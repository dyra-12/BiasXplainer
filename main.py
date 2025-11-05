import os
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import time
import json
from typing import Dict, List, Any
import re

from core.bias_detector import BiasDetector
from core.explainer import SHAPExplainer
from core.counterfactuals import CounterfactualGenerator
from export.json_export import export_results_to_json, save_results_json
from export.csv_export import export_results_to_csv, save_results_csv

class BiasGuardPro:
    def __init__(self, model_path: str = None):
        print("🚀 Initializing BiasGuard Pro...")
        
        if model_path is None:
            model_path = self._auto_detect_model_path()
        
        print(f"📁 Using model path: {model_path}")
        self.detector = BiasDetector(model_path)
        self.explainer = SHAPExplainer(model_path)
        self.counterfactuals = CounterfactualGenerator()
        print("✅ BiasGuard Pro initialized successfully!\n")
    
    def _auto_detect_model_path(self) -> str:
        possible_paths = ['.', './models', './model']
        model_extensions = ('.safetensors', '.bin', '.json')
        for path in possible_paths:
            if os.path.exists(path):
                files = os.listdir(path)
                if any(f.endswith(model_extensions) for f in files):
                    print(f"✅ Found model files in: {path}")
                    return path
        print("⚠️  No local model files found. Using default model...")
        return "distilbert-base-uncased"
    
    def analyze_text(self, text: str) -> Dict:
        print(f"🔍 Analyzing: '{text}'")
        timings = {}
        t0 = time.perf_counter()

        # Run bias prediction and SHAP explainer in parallel to overlap latency
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as exc:
            fut_bias = exc.submit(self.detector.predict_bias, text)
            fut_shap = exc.submit(self.explainer.get_shap_values, text)

            t_start = time.perf_counter()
            bias_result = fut_bias.result()
            timings['predict_bias'] = time.perf_counter() - t_start

            t_start = time.perf_counter()
            shap_results = fut_shap.result()
            timings['get_shap_values'] = time.perf_counter() - t_start

        print(f"   Top biased words: {[w for w, s in shap_results[:3]]}")

        t_start = time.perf_counter()
        counterfactuals = self.counterfactuals.generate_counterfactuals(text, shap_results)
        timings['generate_counterfactuals'] = time.perf_counter() - t_start

        timings['analyze_text_total'] = time.perf_counter() - t0

        return {
            'text': text,
            'bias_probability': bias_result.get('bias_probability'),
            'bias_class': bias_result.get('classification'),
            'confidence': bias_result.get('confidence'),
            'top_biased_words': [w for w, s in shap_results[:3]],
            'shap_scores': shap_results[:10],
            'counterfactuals': counterfactuals,
            'timestamp': time.time(),
            'timings': timings
        }


class BiasGuardDashboard:
    def __init__(self):
        print("🎨 Initializing BiasGuard Pro Dashboard...")
        self.analyzer = BiasGuardPro()
        self.analysis_history = []
        self.last_batch_results: List[Dict] = []
        self._jobs: Dict[str, Dict] = {}
        
        self.sample_texts = [
            "Women should be nurses because they are compassionate.",
            "Men are naturally better at engineering roles.",
            "The female secretary was very emotional today.",
        ]
        
        print("✅ Dashboard initialized successfully!")
    
    def create_bias_meter(self, bias_prob: float) -> go.Figure:
        if bias_prob > 0.7:
            bias_color = "#dc2626"
            bias_label = "HIGH BIAS"
        elif bias_prob > 0.4:
            bias_color = "#f59e0b"
            bias_label = "MODERATE BIAS"
        else:
            bias_color = "#10b981"
            bias_label = "LOW BIAS"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bias_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"<b>{bias_label}</b>", 'font': {'size': 26, 'color': bias_color}},
            number = {'font': {'size': 52, 'color': bias_color}, 'valueformat': '.2f'},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 2, 'tickcolor': "#e5e7eb", 'tickfont': {'size': 14}},
                'bar': {'color': bias_color, 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 3,
                'bordercolor': "#e5e7eb",
                'steps': [
                    {'range': [0, 0.4], 'color': '#d1fae5'},
                    {'range': [0.4, 0.7], 'color': '#fef3c7'},
                    {'range': [0.7, 1], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': bias_color, 'width': 5},
                    'thickness': 0.8,
                    'value': bias_prob
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=80, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font={'family': 'Inter, system-ui, sans-serif', 'size': 14}
        )
        return fig
    
    def create_shap_chart(self, shap_scores: List) -> go.Figure:
        if not shap_scores:
            fig = go.Figure()
            fig.add_annotation(
                text="<b>No significant biased words detected</b>",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=18, color='#64748b')
            )
            fig.update_layout(height=400, paper_bgcolor='white', plot_bgcolor='white')
            return fig
            
        words = [word for word, score in shap_scores[:8]]
        scores = [score for word, score in shap_scores[:8]]
        
        colors = ['#dc2626' if s > 0.1 else '#f59e0b' if s > 0 else '#10b981' for s in scores]
        
        fig = go.Figure(go.Bar(
            x=scores,
            y=words,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=0)
            ),
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "<b>Word Impact Analysis</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#1e293b'}
            },
            xaxis=dict(
                title="<b>SHAP Score</b>",
                showgrid=True,
                gridwidth=1,
                gridcolor='#f1f5f9',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='#cbd5e1'
            ),
            yaxis=dict(
                title="",
                showgrid=False
            ),
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=120, r=30, t=60, b=50),
            font={'family': 'Inter, system-ui, sans-serif', 'size': 14}
        )
        
        return fig
    
    def highlight_biased_words(self, text: str, shap_scores: List) -> str:
        if not shap_scores:
            return f"""
            <div style='padding: 24px; background: white; border-radius: 16px; border: 2px solid #e2e8f0; font-size: 16px; line-height: 1.8; color: #334155;'>
                {text}
            </div>
            """
        
        word_scores = {word.lower(): abs(score) for word, score in shap_scores if score > 0}
        words = text.split()
        highlighted_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in word_scores:
                score = word_scores[clean_word]
                if score > 0.15:
                    color = "#dc2626"
                    label = "High"
                elif score > 0.05:
                    color = "#f59e0b"
                    label = "Medium"
                else:
                    color = "#fbbf24"
                    label = "Low"
                
                highlighted_words.append(
                    f"<mark style='background: {color}; color: white; padding: 4px 10px; border-radius: 8px; font-weight: 600; margin: 0 2px;' title='{label} impact'>{word}</mark>"
                )
            else:
                highlighted_words.append(word)
        
        highlighted_text = " ".join(highlighted_words)
        return f"""
        <div style='padding: 28px; background: white; border-radius: 20px; border: 2px solid #e2e8f0; font-size: 17px; line-height: 2; color: #1e293b; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);'>
            {highlighted_text}
        </div>
        """
    
    def analyze_text_for_dashboard(self, text: str, progress=gr.Progress()) -> Dict[str, Any]:
        if not text.strip():
            return {"error": "Please enter text to analyze"}
        
        try:
            # Show progress
            ui_timings = {}
            progress(0.2, desc="Detecting bias...")
            t_sleep = time.perf_counter()
            time.sleep(0.3)
            ui_timings['progress_sleep_1'] = time.perf_counter() - t_sleep

            result = self.analyzer.analyze_text(text)

            progress(0.6, desc="Analyzing word impact...")
            t_sleep = time.perf_counter()
            time.sleep(0.3)
            ui_timings['progress_sleep_2'] = time.perf_counter() - t_sleep
            
            self.analysis_history.append(result)
            if len(self.analysis_history) > 10:
                self.analysis_history.pop(0)
            
            progress(0.8, desc="Generating alternatives...")
            t_sleep = time.perf_counter()
            time.sleep(0.2)
            ui_timings['progress_sleep_3'] = time.perf_counter() - t_sleep

            t_start = time.perf_counter()
            bias_meter = self.create_bias_meter(result['bias_probability'])
            ui_timings['create_bias_meter'] = time.perf_counter() - t_start

            t_start = time.perf_counter()
            shap_chart = self.create_shap_chart(result['shap_scores'])
            ui_timings['create_shap_chart'] = time.perf_counter() - t_start

            t_start = time.perf_counter()
            highlighted_text = self.highlight_biased_words(text, result['shap_scores'])
            ui_timings['highlight_biased_words'] = time.perf_counter() - t_start
            
            # Create summary card with fixed font colors
            bias_level = "highly biased" if result['bias_probability'] > 0.7 else \
                       "moderately biased" if result['bias_probability'] > 0.4 else "relatively neutral"
            
            if result['bias_probability'] > 0.7:
                summary_gradient = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
                border_color = "#fca5a5"
                text_color = "#991b1b"
                emoji = "🚨"
            elif result['bias_probability'] > 0.4:
                summary_gradient = "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
                border_color = "#fcd34d"
                text_color = "#92400e"
                emoji = "⚠️"
            else:
                summary_gradient = "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
                border_color = "#6ee7b7"
                text_color = "#065f46"
                emoji = "✅"
            
            summary_html = f"""
            <div style='padding: 32px; background: {summary_gradient}; border-radius: 24px; border: 3px solid {border_color}; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);'>
                <div style='display: flex; align-items: center; gap: 16px; margin-bottom: 20px;'>
                    <span style='font-size: 48px;'>{emoji}</span>
                    <div>
                        <h3 style='margin: 0; color: {text_color}; font-size: 24px; font-weight: 800;'>Analysis Complete</h3>
                        <p style='margin: 4px 0 0 0; color: {text_color}; font-size: 16px; opacity: 0.8;'>Bias detection results</p>
                    </div>
                </div>
                <div style='background: white; padding: 20px; border-radius: 16px; margin-bottom: 16px;'>
                    <div style='font-size: 16px; color: #334155; margin-bottom: 12px;'>
                        This text is <strong style='color: {text_color}; font-size: 18px;'>{bias_level}</strong>
                    </div>
                    <div style='display: flex; align-items: center; gap: 12px;'>
                        <span style='font-size: 14px; color: #64748b; font-weight: 600;'>Bias Score:</span>
                        <span style='font-size: 32px; font-weight: 900; color: {text_color};'>{result['bias_probability']:.3f}</span>
                    </div>
                </div>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 14px;'>
                    <div style='background: rgba(255,255,255,0.7); padding: 12px; border-radius: 12px;'>
                        <strong style='color: #1e293b;'>Classification:</strong> <span style='color: {text_color}; font-weight: 600;'>{result['bias_class']}</span>
                    </div>
                    <div style='background: rgba(255,255,255,0.7); padding: 12px; border-radius: 12px;'>
                        <strong style='color: #1e293b;'>Confidence:</strong> <span style='color: {text_color}; font-weight: 600;'>{result['confidence']:.1%}</span>
                    </div>
                </div>
            </div>
            """
            
            # Create counterfactuals HTML
            counterfactuals_html = "<div style='margin: 20px 0;'>"
            if result['counterfactuals']:
                t_start = time.perf_counter()
                for i, cf in enumerate(result['counterfactuals'], 1):
                    counterfactuals_html += f"""
                    <div style='margin: 20px 0; padding: 24px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-left: 5px solid #10b981; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);'>
                        <div style='font-size: 13px; color: #047857; font-weight: 700; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em;'>
                            ✨ Alternative {i}
                        </div>
                        <div style='font-size: 17px; color: #065f46; line-height: 1.7; font-weight: 500;'>{cf}</div>
                    </div>
                    """
                ui_timings['build_counterfactuals_html'] = time.perf_counter() - t_start
            else:
                counterfactuals_html += """
                <div style='padding: 32px; background: #f8fafc; border-radius: 20px; text-align: center; border: 3px dashed #cbd5e1;'>
                    <div style='font-size: 56px; margin-bottom: 16px;'>💡</div>
                    <div style='font-size: 16px; color: #64748b; font-weight: 600;'>No alternative suggestions available</div>
                </div>
                """
            counterfactuals_html += "</div>"
            
            # Top words display
            top_words_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px;'>"
            t_start = time.perf_counter()
            for word in result['top_biased_words']:
                top_words_html += f"""
                <span style='padding: 8px 16px; background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b; border-radius: 12px; font-weight: 700; font-size: 14px; border: 2px solid #fca5a5;'>
                    "{word}"
                </span>
                """
            ui_timings['build_top_words_html'] = time.perf_counter() - t_start
            top_words_html += "</div>" if result['top_biased_words'] else "<div style='color: #94a3b8; font-style: italic;'>None detected</div>"
            
            progress(1.0, desc="Complete!")

            # Merge analyzer timings and ui timings for a quick profile
            profiler = {}
            if isinstance(result, dict) and 'timings' in result:
                profiler.update({f"analyzer.{k}": v for k, v in result['timings'].items()})
            profiler.update({f"ui.{k}": v for k, v in ui_timings.items()})

            # Print a sorted timing report to console
            try:
                sorted_times = sorted(profiler.items(), key=lambda x: x[1], reverse=True)
                print("\n📈 Profiling report (descending):")
                for name, dur in sorted_times:
                    print(f" - {name}: {dur:.4f}s")
            except Exception:
                sorted_times = []

            # Small HTML snippet to show top 5 timings in the UI
            profiling_html = "<div style='margin-top:12px; padding: 12px; border-radius: 12px; background: #f8fafc; border: 2px solid #e2e8f0; font-size:13px;'>"
            profiling_html += "<strong>⏱️ Timing Breakdown:</strong><br/>"
            try:
                for name, dur in sorted_times[:5]:
                    profiling_html += f"<div style='display:flex; justify-content:space-between; gap:12px;'><span style='color:#334155;'>{name}</span><span style='font-weight:800;'>{dur:.3f}s</span></div>"
            except Exception:
                profiling_html += "<div>Profiling not available</div>"
            profiling_html += "</div>"

            return {
                "success": True,
                "bias_meter": bias_meter,
                "shap_chart": shap_chart,
                "highlighted_text": highlighted_text,
                "summary_html": summary_html,
                "counterfactuals_html": counterfactuals_html,
                "top_words_html": top_words_html,
                "bias_probability": result['bias_probability'],
                "bias_class": result['bias_class'],
                "profiling_html": profiling_html,
                "profiling_data": profiler
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def create_dashboard(self):
        custom_css = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
            --background-fill-primary: #ffffff !important;
            --body-background-fill: var(--background-fill-primary) !important;
            --background-fill-secondary: #ffffff !important;
            --app-background: #ffffff !important;
        }

        html, body, .gradio-root, .gradio-container {
            background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 50%, #f3e8ff 100%) !important;
            min-height: 100vh !important;
            color: #1e293b !important;
        }

        :root .dark, .gradio-root.dark, .dark, .dark .gradio-root, .dark .gradio-container, [data-theme="dark"] {
            --background-fill-primary: #ffffff !important;
            --body-background-fill: #ffffff !important;
            --background-fill-secondary: #ffffff !important;
            --app-background: #ffffff !important;
            background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 50%, #f3e8ff 100%) !important;
            color: #1e293b !important;
        }

        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }
        
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto !important;
            background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 50%, #f3e8ff 100%) !important;
        }
        
        .input-textarea, .output-textarea, textarea {
            background: white !important;
            color: #1e293b !important;
        }
        
        .prose {
            color: #1e293b !important;
        }
        
        .main-header {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%);
            padding: 48px 40px;
            border-radius: 28px;
            margin-bottom: 32px;
            box-shadow: 0 20px 60px rgba(124, 58, 237, 0.4);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 80%, rgba(255,255,255,0.1) 0%, transparent 50%);
            opacity: 0.6;
        }
        
        .input-card {
            background: white !important;
            border-radius: 24px;
            padding: 32px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 2px solid #e2e8f0;
        }
        
        textarea, .input-textarea textarea {
            border-radius: 16px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 18px !important;
            font-size: 15px !important;
            line-height: 1.7 !important;
            transition: all 0.3s ease !important;
            background: white !important;
            color: #1e293b !important;
        }
        
        textarea:focus, .input-textarea textarea:focus {
            border-color: #7c3aed !important;
            box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.1) !important;
            background: white !important;
        }
        
        textarea::placeholder {
            color: #94a3b8 !important;
        }
        
        /* Enhanced Sample Button Styling */
        .sample-btn, button.sample-btn {
            background: linear-gradient(135deg, #ede9fe, #ddd6fe) !important;
            color: #5b21b6 !important;
            border: 2px solid #c4b5fd !important;
            border-radius: 12px !important;
            padding: 10px 18px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            transition: all 0.3s ease !important;
            min-width: 220px !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            text-align: left !important;
            line-height: 1.4 !important;
            height: auto !important;
            min-height: 48px !important;
        }
        
        .sample-btn:hover, button.sample-btn:hover {
            background: linear-gradient(135deg, #ddd6fe, #c4b5fd) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3) !important;
        }
        
        .primary-btn, button.primary-btn {
            background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
            color: white !important;
            border: none !important;
            border-radius: 18px !important;
            padding: 16px 40px !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 10px 25px rgba(124, 58, 237, 0.4) !important;
        }
        
        .primary-btn:hover, button.primary-btn:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 15px 35px rgba(124, 58, 237, 0.5) !important;
        }
        
        .secondary-btn, button.secondary-btn {
            background: #f1f5f9 !important;
            color: #475569 !important;
            border: 2px solid #cbd5e1 !important;
            border-radius: 14px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .secondary-btn:hover, button.secondary-btn:hover {
            background: #e2e8f0 !important;
            border-color: #94a3b8 !important;
        }
        
        .info-card {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-radius: 24px;
            padding: 28px;
            border: 3px solid #fcd34d;
            box-shadow: 0 8px 20px rgba(245, 158, 11, 0.25);
        }
        
        .step-card {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 16px;
            margin: 12px 0;
            border: 2px solid rgba(255,255,255,0.5);
        }
        
        .results-section {
            animation: slideUp 0.6s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .tab-nav button {
            border-radius: 14px 14px 0 0 !important;
            font-weight: 600 !important;
            padding: 14px 28px !important;
            transition: all 0.3s ease !important;
            border: 2px solid transparent !important;
            font-size: 15px !important;
            background: #f8fafc !important;
            color: #475569 !important;
        }
        
        .tab-nav button[aria-selected="true"] {
            background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
            color: white !important;
            border-bottom: 2px solid #7c3aed !important;
        }
        
        .tab-nav button:hover {
            background: #e2e8f0 !important;
            color: #1e293b !important;
        }
        
        .tab-nav button[aria-selected="true"]:hover {
            background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
            color: white !important;
        }
        
        .tabitem {
            background: white !important;
            border-radius: 0 0 20px 20px !important;
            padding: 24px !important;
        }
        
        .result-card {
            background: white;
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 2px solid #e2e8f0;
            margin: 16px 0;
        }
        
        .plot-container {
            background: white !important;
            border-radius: 20px !important;
            padding: 16px !important;
        }

        .block.svelte-1svsvh2, .svelte-1svsvh2, .gradio-block, .gradio-block * {
            background: white !important;
            color: #1e293b !important;
        }

        .block.svelte-1svsvh2 {
            position: relative !important;
            margin: 0 !important;
            box-shadow: var(--block-shadow, 0 4px 20px rgba(0,0,0,0.06)) !important;
            border-width: var(--block-border-width, 1px) !important;
            border-color: var(--block-border-color, #e2e8f0) !important;
            border-radius: var(--block-radius, 12px) !important;
            background: #ffffff !important;
            width: 100% !important;
            line-height: var(--line-sm, 1.2) !important;
        }

        button.svelte-i00v67.svelte-i00v67,
        button.svelte-i00v67,
        .gr-button, .gr-button *,
        .sample-btn, button.sample-btn {
            color: #0f172a !important;
            text-shadow: none !important;
        }

        .input-card, .input-card * {
            background: white !important;
            color: #0f172a !important;
        }
        
        /* Enhanced Batch Analysis Tab */
        .batch-analysis-card {
            background: white !important;
            border-radius: 24px;
            padding: 0px;
        }
        
        .batch-header {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 32px;
            border: 2px solid #cbd5e1;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        
        .batch-section {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            border: 2px solid #cbd5e1;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        }
        
        .batch-section:hover {
            border-color: #94a3b8;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            transition: all 0.3s ease;
        }
        
        .upload-zone {
            background: linear-gradient(135deg, #fafafa, #f5f5f5) !important;
            border: 2px dashed #94a3b8 !important;
            border-radius: 16px !important;
            padding: 32px !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            cursor: pointer !important;
        }
        
        .upload-zone:hover {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe) !important;
            border-color: #60a5fa !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15) !important;
        }
        
        .progress-card {
            background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
            border-radius: 16px;
            padding: 24px;
            border: 2px solid #cbd5e1;
            margin-bottom: 16px;
        }
        
        /* Batch Analysis Input Fields Styling */
        #batch-tab .gr-box,
        #batch-tab .gr-form,
        #batch-tab input[type="number"],
        #batch-tab input[type="text"],
        #batch-tab textarea {
            background: white !important;
            border: 2px solid #cbd5e1 !important;
            border-radius: 12px !important;
            padding: 14px 16px !important;
            font-size: 14px !important;
            color: #1e293b !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        #batch-tab input[type="number"]:focus,
        #batch-tab input[type="text"]:focus,
        #batch-tab textarea:focus {
            background: white !important;
            border-color: #7c3aed !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
            outline: none !important;
        }
        
        #batch-tab input[type="number"]:disabled,
        #batch-tab input[type="text"]:disabled,
        #batch-tab textarea:disabled {
            background: #f8fafc !important;
            color: #64748b !important;
            border-color: #cbd5e1 !important;
            cursor: not-allowed !important;
            font-weight: 600 !important;
        }
        
        #batch-tab input::placeholder,
        #batch-tab textarea::placeholder {
            color: #94a3b8 !important;
            font-weight: 400 !important;
        }
        
        #batch-tab label {
            font-weight: 700 !important;
            color: #1e293b !important;
            font-size: 14px !important;
            margin-bottom: 8px !important;
            letter-spacing: 0.01em !important;
        }
        
        #batch-tab .gr-file {
            background: white !important;
            border: 2px solid #cbd5e1 !important;
            border-radius: 12px !important;
            padding: 12px !important;
        }
        
        /* Progress and Export Input Groups */
        .progress-inputs,
        .export-inputs {
            background: #f8fafc;
            border-radius: 14px;
            padding: 20px;
            border: 2px solid #cbd5e1;
            margin-top: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .progress-inputs > *,
        .export-inputs > * {
            margin-bottom: 12px;
        }
        
        .progress-inputs > *:last-child,
        .export-inputs > *:last-child {
            margin-bottom: 0;
        }
        
        .export-section {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-radius: 16px;
            padding: 20px;
            border: 2px solid #fbbf24;
        }
        
        .batch-input-section {
            background: #f8fafc;
            border-radius: 16px;
            padding: 24px;
            border: 2px solid #cbd5e1;
            margin-bottom: 20px;
        }
        
        /* Section Headers in Batch Tab */
        .batch-section-header {
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
            border-left: 4px solid #7c3aed;
            border: 2px solid #cbd5e1;
            border-left: 4px solid #7c3aed;
        }
        
        .batch-section-header h4 {
            margin: 0;
            font-size: 16px;
            font-weight: 800;
            color: #1e293b;
        }
        
        .batch-section-header p {
            margin: 4px 0 0 0;
            font-size: 13px;
            color: #64748b;
        }
        
        .legend-item {
            display: inline-flex;
            align-items: center;
            margin: 0 12px;
            padding: 6px 14px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 13px;
        }
        
        .footer {
            margin-top: 48px;
            padding: 32px;
            background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
            border-radius: 24px;
            text-align: center;
        }
        
        .footer a {
            color: #7c3aed !important;
            text-decoration: none !important;
            font-weight: 600 !important;
            transition: color 0.3s ease !important;
        }
        
        .footer a:hover {
            color: #5b21b6 !important;
        }
        
        input[type="number"] {
            background: white !important;
            color: #1e293b !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 12px !important;
            padding: 12px !important;
        }
        
        .input-textbox input, input[type="text"] {
            background: white !important;
            color: #1e293b !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 12px !important;
            padding: 12px !important;
        }
        
        .input-textbox input:focus, input[type="text"]:focus {
            border-color: #7c3aed !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
        }
        
        label {
            color: #334155 !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        /* Smooth scroll behavior */
        html {
            scroll-behavior: smooth;
        }
        
        /* Loading spinner animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        """
        
        with gr.Blocks(css=custom_css, title="BiasGuard Pro", theme=gr.themes.Default()) as demo:
            
            # Header
            gr.HTML("""
            <div class='main-header'>
                <div style='position: relative; z-index: 1;'>
                    <div style='display: flex; align-items: center; gap: 20px; margin-bottom: 12px;'>
                        <div style='background: rgba(255,255,255,0.2); backdrop-filter: blur(10px); padding: 16px; border-radius: 20px;'>
                            <span style='font-size: 48px;'>🛡️</span>
                        </div>
                        <div>
                            <h1 style='margin: 0; font-size: 56px; font-weight: 900; color: white; text-shadow: 0 4px 12px rgba(0,0,0,0.2); letter-spacing: -0.02em;'>
                                BiasGuard Pro
                            </h1>
                            <p style='margin: 8px 0 0 0; font-weight: 500; color: rgba(255,255,255,0.95); font-size: 20px;'>
                                AI-Powered Bias Detection & Mitigation
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            """)
            
            with gr.Row(equal_height=False):
                # Main Content Area (Left)
                with gr.Column(scale=2):
                    with gr.Group(elem_classes="input-card"):
                        gr.HTML("""
                        <div style='margin-bottom: 20px;'>
                            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
                                <div style='background: linear-gradient(135deg, #3b82f6, #06b6d4); padding: 10px; border-radius: 12px;'>
                                    <span style='font-size: 20px;'>✨</span>
                                </div>
                                <h2 style='margin: 0; font-size: 24px; font-weight: 800; color: #1e293b;'>Analyze Text</h2>
                            </div>
                            <p style='margin: 0; color: #64748b; font-size: 14px;'>Enter or paste text to detect gender bias</p>
                        </div>
                        """)
                        
                        text_input = gr.Textbox(
                            label="",
                            placeholder="Example: 'Women should be nurses because they are compassionate and caring...'",
                            lines=6,
                            max_lines=10,
                            show_label=False
                        )
                        
                        gr.HTML("<div style='margin: 20px 0 12px 0; font-size: 13px; font-weight: 700; color: #475569;'>💡 Quick Examples:</div>")
                        with gr.Row():
                            sample_btn_1 = gr.Button(
                                self.sample_texts[0],
                                size="sm",
                                elem_classes="sample-btn"
                            )
                            sample_btn_2 = gr.Button(
                                self.sample_texts[1],
                                size="sm",
                                elem_classes="sample-btn"
                            )
                            sample_btn_3 = gr.Button(
                                self.sample_texts[2],
                                size="sm",
                                elem_classes="sample-btn"
                            )
                        
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "🔍 Analyze for Bias",
                                variant="primary",
                                size="lg",
                                elem_classes="primary-btn",
                                scale=3
                            )
                            clear_btn = gr.Button(
                                "Clear", 
                                size="lg",
                                elem_classes="secondary-btn",
                                scale=1
                            )
                
                # Sidebar (Right)
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class='info-card'>
                        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 20px;'>
                            <span style='font-size: 32px;'>⚡</span>
                            <h3 style='margin: 0; color: #78350f; font-size: 22px; font-weight: 800;'>How It Works</h3>
                        </div>
                        
                        <div class='step-card'>
                            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
                                <div style='min-width: 36px; width: 36px; height: 36px; background: linear-gradient(135deg, #3b82f6, #06b6d4); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 16px;'>1</div>
                                <h4 style='margin: 0; font-size: 16px; font-weight: 700; color: #1e293b;'>Detection</h4>
                            </div>
                            <p style='margin: 0; font-size: 13px; color: #475569; line-height: 1.6;'>AI identifies gender stereotypes and biased language patterns</p>
                        </div>

                        <div class='step-card'>
                            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
                                <div style='min-width: 36px; width: 36px; height: 36px; background: linear-gradient(135deg, #a855f7, #ec4899); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 16px;'>2</div>
                                <h4 style='margin: 0; font-size: 16px; font-weight: 700; color: #1e293b;'>Explanation</h4>
                            </div>
                            <p style='margin: 0; font-size: 13px; color: #475569; line-height: 1.6;'>SHAP highlights problematic words and phrases</p>
                        </div>

                        <div class='step-card'>
                            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
                                <div style='min-width: 36px; width: 36px; height: 36px; background: linear-gradient(135deg, #10b981, #14b8a6); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 16px;'>3</div>
                                <h4 style='margin: 0; font-size: 16px; font-weight: 700; color: #1e293b;'>Mitigation</h4>
                            </div>
                            <p style='margin: 0; font-size: 13px; color: #475569; line-height: 1.6;'>Generates neutral, inclusive alternatives</p>
                        </div>

                        <div style='margin-top: 20px; padding-top: 16px; border-top: 3px solid #fcd34d;'>
                            <p style='margin: 0; font-size: 13px; font-weight: 700; color: #78350f; display: flex; align-items: center; gap: 8px;'>
                                <span style='font-size: 18px;'>✨</span>
                                Powered by DistilBERT + SHAP
                            </p>
                        </div>
                    </div>
                    """)
            
            # Scroll anchor for results
            results_anchor = gr.HTML("<div id='results-anchor'></div>", visible=False)
            
            # Results Section
            with gr.Column(visible=False, elem_classes="results-section") as results_section:
                gr.HTML("<div style='margin: 32px 0 24px 0;'><h2 style='font-size: 32px; font-weight: 900; color: #1e293b; margin: 0;'>📊 Analysis Results</h2></div>")
                
                # Summary and Bias Meter Row
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        bias_meter = gr.Plot(label="")
                    
                    with gr.Column(scale=2):
                        summary_display = gr.HTML(label="")
                
                # Detailed Analysis Tabs
                gr.HTML("<div style='margin: 32px 0 16px 0;'><h3 style='font-size: 24px; font-weight: 800; color: #1e293b; margin: 0;'>🔍 Detailed Analysis</h3></div>")
                
                with gr.Tabs() as tabs:
                    with gr.TabItem("📊 Word Impact", id="word-tab"):
                        gr.HTML("""
                        <div style='background: linear-gradient(135deg, #f8fafc, #f1f5f9); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; border-left: 4px solid #3b82f6;'>
                            <div style='display: flex; align-items: center; gap: 10px;'>
                                <span style='font-size: 24px;'>📊</span>
                                <div>
                                    <h4 style='margin: 0; font-size: 16px; font-weight: 800; color: #1e293b;'>Word Impact Analysis</h4>
                                    <p style='margin: 4px 0 0 0; font-size: 13px; color: #64748b;'>SHAP values showing which words contribute most to bias detection</p>
                                </div>
                            </div>
                        </div>
                        """)
                        with gr.Row():
                            with gr.Column():
                                shap_chart = gr.Plot(label="")
                        
                        gr.HTML("<div style='margin: 24px 0 12px 0;'><h4 style='font-size: 18px; font-weight: 700; color: #334155; margin: 0;'>🎯 Key Biased Terms</h4></div>")
                        top_words_display = gr.HTML(label="")
                    
                    with gr.TabItem("📝 Highlighted Text", id="text-tab"):
                        gr.HTML("""
                        <div style='background: linear-gradient(135deg, #f8fafc, #f1f5f9); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; border-left: 4px solid #f59e0b;'>
                            <div style='display: flex; align-items: center; gap: 10px;'>
                                <span style='font-size: 24px;'>📝</span>
                                <div>
                                    <h4 style='margin: 0; font-size: 16px; font-weight: 800; color: #1e293b;'>Original Text with Bias Indicators</h4>
                                    <p style='margin: 4px 0 0 0; font-size: 13px; color: #64748b;'>Words are highlighted based on their contribution to bias</p>
                                </div>
                            </div>
                        </div>
                        """)
                        highlighted_text = gr.HTML(label="")
                        
                        gr.HTML("""
                        <div style='margin-top: 24px; padding: 20px; background: linear-gradient(135deg, #eff6ff, #dbeafe); border-radius: 16px; border-left: 5px solid #3b82f6;'>
                            <strong style='color: #1e40af; font-size: 15px; display: block; margin-bottom: 12px;'>Legend:</strong>
                            <div style='display: flex; flex-wrap: wrap; gap: 12px;'>
                                <span class='legend-item' style='background: #dc2626; color: white;'>High Impact</span>
                                <span class='legend-item' style='background: #f59e0b; color: white;'>Medium Impact</span>
                                <span class='legend-item' style='background: #fbbf24; color: white;'>Low Impact</span>
                            </div>
                        </div>
                        """)
                    
                    with gr.TabItem("🔄 Neutral Alternatives", id="alternatives-tab"):
                        gr.HTML("""
                        <div style='background: linear-gradient(135deg, #f8fafc, #f1f5f9); border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; border-left: 4px solid #10b981;'>
                            <div style='display: flex; align-items: center; gap: 10px;'>
                                <span style='font-size: 24px;'>🔄</span>
                                <div>
                                    <h4 style='margin: 0; font-size: 16px; font-weight: 800; color: #1e293b;'>AI-Generated Suggestions</h4>
                                    <p style='margin: 4px 0 0 0; font-size: 13px; color: #64748b;'>Bias-free alternatives that maintain your core message</p>
                                </div>
                            </div>
                        </div>
                        """)
                        counterfactuals_display = gr.HTML(label="")
                        
                        gr.HTML("""
                        <div style='margin-top: 24px; padding: 24px; background: linear-gradient(135deg, #e0f2fe, #bae6fd); border-radius: 20px; border-left: 5px solid #0284c7;'>
                            <strong style='color: #075985; font-size: 16px; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
                                <span style='font-size: 24px;'>💡</span>
                                Pro Tip
                            </strong>
                            <p style='margin: 0; color: #0c4a6e; font-size: 14px; line-height: 1.7;'>
                                These alternatives maintain your core message while removing gendered stereotypes. 
                                Use them to create more inclusive career recommendations.
                            </p>
                        </div>
                        """)
                    
                    with gr.TabItem("📚 Batch Analysis", id="batch-tab", elem_classes="batch-analysis-card"):
                        gr.HTML("""
                        <div class='batch-header'>
                            <div style='text-align: center;'>
                                <div style='display: inline-flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #7c3aed, #a855f7); width: 64px; height: 64px; border-radius: 20px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);'>
                                    <span style='font-size: 36px;'>📦</span>
                                </div>
                                <h3 style='font-size: 28px; font-weight: 900; color: #1e293b; margin: 0 0 8px 0;'>Batch Processing</h3>
                                <p style='margin: 0; font-size: 15px; color: #64748b; font-weight: 500;'>Process multiple texts simultaneously and compare group statistics</p>
                            </div>
                        </div>
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                # Input Section
                                gr.HTML("""
                                <div class='batch-section-header'>
                                    <div style='display: flex; align-items: center; gap: 10px;'>
                                        <span style='font-size: 24px;'>📝</span>
                                        <div>
                                            <h4 style='margin: 0; font-size: 16px; font-weight: 800; color: #1e293b;'>Text Input</h4>
                                            <p style='margin: 4px 0 0 0; font-size: 13px; color: #64748b;'>Enter or paste your texts below (one per line)</p>
                                        </div>
                                    </div>
                                </div>
                                """)
                                
                                batch_textarea = gr.Textbox(
                                    label="",
                                    lines=8,
                                    placeholder="Paste your texts here, one per line...\n\nExample:\nWomen are naturally better at nursing.\nMen excel in technical fields.\nThe female assistant was emotional.",
                                    elem_classes="batch-input"
                                )
                                
                                gr.HTML("""
                                <div style='margin: 24px 0; padding: 16px; background: #f8fafc; border-radius: 12px; text-align: center; border: 2px dashed #94a3b8;'>
                                    <div style='display: inline-flex; align-items: center; gap: 10px;'>
                                        <span style='font-size: 20px;'>📁</span>
                                        <span style='font-weight: 700; color: #64748b; font-size: 14px;'>OR UPLOAD A FILE</span>
                                        <span style='font-size: 20px;'>↓</span>
                                    </div>
                                </div>
                                """)
                                
                                file_upload = gr.File(
                                    label="",
                                    file_types=[".txt", ".csv", ".json"],
                                    elem_classes="upload-zone"
                                )
                                
                                # Group Comparison Section
                                gr.HTML("""
                                <div class='batch-section-header' style='border-left-color: #f59e0b;'>
                                    <div style='display: flex; align-items: center; gap: 10px;'>
                                        <span style='font-size: 24px;'>🎯</span>
                                        <div>
                                            <h4 style='margin: 0; font-size: 16px; font-weight: 800; color: #1e293b;'>Group Comparison</h4>
                                            <p style='margin: 4px 0 0 0; font-size: 13px; color: #64748b;'>Optional: Filter texts into groups for comparative analysis</p>
                                        </div>
                                    </div>
                                </div>
                                """)
                                
                                with gr.Row():
                                    group_a_select = gr.Textbox(
                                        label="🔵 Group A Filter",
                                        placeholder="e.g., 'female' or 'women'",
                                        scale=1
                                    )
                                    group_b_select = gr.Textbox(
                                        label="🔴 Group B Filter",
                                        placeholder="e.g., 'male' or 'men'",
                                        scale=1
                                    )
                                
                                # Action Buttons
                                gr.HTML("<div style='height: 20px;'></div>")
                                with gr.Row():
                                    run_batch_btn = gr.Button(
                                        "🚀 Start Batch Processing",
                                        variant="primary",
                                        size="lg",
                                        elem_classes="primary-btn",
                                        scale=2
                                    )
                                    refresh_status_btn = gr.Button(
                                        "🔄 Refresh Status",
                                        size="lg",
                                        elem_classes="secondary-btn",
                                        scale=1
                                    )
                            
                            with gr.Column(scale=1):
                                # Progress Monitor
                                gr.HTML("""
                                <div class='batch-section-header' style='border-left-color: #3b82f6; margin-bottom: 16px;'>
                                    <div style='display: flex; align-items: center; gap: 10px;'>
                                        <span style='font-size: 24px;'>⏱️</span>
                                        <div>
                                            <h4 style='font-size: 16px; font-weight: 800; color: #1e293b; margin: 0;'>Progress Monitor</h4>
                                            <p style='margin: 4px 0 0 0; font-size: 13px; color: #64748b;'>Track your batch processing status in real-time</p>
                                        </div>
                                    </div>
                                </div>
                                """)
                                
                                with gr.Group(elem_classes="progress-inputs"):
                                    progress_bar = gr.Number(
                                        value=0,
                                        label="📊 Completion Percentage",
                                        precision=1,
                                        interactive=False
                                    )
                                    progress_text = gr.Textbox(
                                        interactive=False,
                                        label="📡 Current Status",
                                        value="Ready to process"
                                    )
                                    job_id_text = gr.Textbox(
                                        interactive=False,
                                        label="🆔 Job ID",
                                        placeholder="Will appear after starting"
                                    )
                                
                                # Export Section
                                gr.HTML("""
                                <div class='batch-section-header' style='border-left-color: #10b981; margin-top: 24px; margin-bottom: 16px;'>
                                    <div style='display: flex; align-items: center; gap: 10px;'>
                                        <span style='font-size: 24px;'>💾</span>
                                        <div>
                                            <h4 style='font-size: 16px; font-weight: 800; color: #1e293b; margin: 0;'>Export Results</h4>
                                            <p style='margin: 4px 0 0 0; font-size: 13px; color: #64748b;'>Save your analysis results</p>
                                        </div>
                                    </div>
                                </div>
                                """)
                                
                                with gr.Group(elem_classes="export-inputs"):
                                    save_path = gr.Textbox(
                                        label="📂 Save Path (optional)",
                                        placeholder="./exports/results.json"
                                    )
                                    with gr.Row():
                                        export_json_btn = gr.Button("📄 Export JSON", size="sm", elem_classes="secondary-btn")
                                        export_csv_btn = gr.Button("📊 Export CSV", size="sm", elem_classes="secondary-btn")
                                    export_download = gr.File(
                                        label="⬇️ Download File",
                                        interactive=False
                                    )
                        
                        # Results Section
                        gr.HTML("""
                        <div class='batch-section-header' style='border-left-color: #10b981; margin: 48px 0 20px 0;'>
                            <div style='display: flex; align-items: center; gap: 10px;'>
                                <span style='font-size: 28px;'>📊</span>
                                <div>
                                    <h4 style='font-size: 20px; font-weight: 900; color: #1e293b; margin: 0;'>Batch Summary & Results</h4>
                                    <p style='margin: 4px 0 0 0; font-size: 13px; color: #64748b;'>Comprehensive analysis of all processed texts</p>
                                </div>
                            </div>
                        </div>
                        """)
                        batch_summary_html = gr.HTML()
                        comparison_html = gr.HTML()
            
            # Loading and Error Display
            loading_display = gr.HTML(visible=False)
            error_display = gr.HTML(visible=False)
            
            # Footer
            gr.HTML("""
            <div class='footer'>
                <h4 style='margin: 0 0 12px 0; color: #1e293b; font-size: 20px; font-weight: 800;'>BiasGuard Pro v1.0</h4>
                <p style='margin: 0 0 20px 0; color: #64748b; font-size: 15px;'>
                    Building fairer AI systems through transparent bias detection and mitigation
                </p>
                <div style='display: flex; justify-content: center; gap: 32px; flex-wrap: wrap;'>
                    <a href='#'>📄 Research Paper</a>
                    <a href='#'>💻 GitHub</a>
                    <a href='#'>📚 Documentation</a>
                    <a href='#'>🤝 Contribute</a>
                </div>
            </div>
            """)
            
            # Event Handlers
            def update_display(text):
                if not text.strip():
                    return {
                        results_section: gr.update(visible=False),
                        loading_display: gr.update(visible=False),
                        error_display: gr.update(
                            value="<div style='padding: 24px; background: linear-gradient(135deg, #fef3c7, #fde68a); border-left: 5px solid #f59e0b; border-radius: 16px; margin: 20px 0;'><strong style='color: #92400e; font-size: 16px;'>⚠️ No Input:</strong> <span style='color: #78350f; font-size: 15px; margin-left: 8px;'>Please enter text to analyze.</span></div>",
                            visible=True
                        ),
                        analyze_btn: gr.update(value="🔍 Analyze for Bias", interactive=True)
                    }
                
                result = self.analyze_text_for_dashboard(text)
                
                if "error" in result:
                    return {
                        results_section: gr.update(visible=False),
                        loading_display: gr.update(visible=False),
                        error_display: gr.update(
                            value=f"<div style='padding: 24px; background: linear-gradient(135deg, #fee2e2, #fecaca); border-left: 5px solid #dc2626; border-radius: 16px; margin: 20px 0;'><strong style='color: #991b1b; font-size: 16px;'>❌ Error:</strong> <span style='color: #7f1d1d; font-size: 15px; margin-left: 8px;'>{result['error']}</span></div>",
                            visible=True
                        ),
                        analyze_btn: gr.update(value="🔍 Analyze for Bias", interactive=True)
                    }
                
                # Auto-scroll to results with improved targeting
                scroll_js = """
                <script>
                setTimeout(function() {
                    // Try multiple selectors to find the results section
                    const resultsSection = document.querySelector('.results-section') || 
                                         document.querySelector('[class*="results-section"]') ||
                                         document.getElementById('results-anchor');
                    if (resultsSection) {
                        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        window.scrollBy(0, -20); // Add slight offset for better visibility
                    }
                }, 600);
                </script>
                """
                
                return {
                    results_section: gr.update(visible=True),
                    loading_display: gr.update(visible=False),
                    error_display: gr.update(visible=False),
                    bias_meter: result['bias_meter'],
                    shap_chart: result['shap_chart'],
                    highlighted_text: result['highlighted_text'],
                    summary_display: result['summary_html'] + scroll_js,
                    counterfactuals_display: result['counterfactuals_html'],
                    top_words_display: result['top_words_html'],
                    analyze_btn: gr.update(value="🔍 Analyze for Bias", interactive=True)
                }
            
            def clear_inputs():
                return ""
            
            def scroll_to_results():
                """Trigger scroll after results are displayed"""
                return None
            
            # Connect analyze button with loading state
            analyze_btn.click(
                lambda: gr.update(value="⏳ Analyzing...", interactive=False),
                inputs=None,
                outputs=[analyze_btn]
            ).then(
                update_display,
                inputs=[text_input],
                outputs=[
                    results_section, loading_display, error_display, bias_meter, shap_chart,
                    highlighted_text, summary_display, counterfactuals_display,
                    top_words_display, analyze_btn
                ],
                scroll_to_output=True
            )
            
            # Connect text input submit with loading state
            text_input.submit(
                lambda: gr.update(value="⏳ Analyzing...", interactive=False),
                inputs=None,
                outputs=[analyze_btn]
            ).then(
                update_display,
                inputs=[text_input],
                outputs=[
                    results_section, loading_display, error_display, bias_meter, shap_chart,
                    highlighted_text, summary_display, counterfactuals_display,
                    top_words_display, analyze_btn
                ],
                scroll_to_output=True
            )
            
            clear_btn.click(clear_inputs, outputs=[text_input])
            
            # Connect sample buttons
            sample_btn_1.click(lambda: self.sample_texts[0], outputs=text_input)
            sample_btn_2.click(lambda: self.sample_texts[1], outputs=text_input)
            sample_btn_3.click(lambda: self.sample_texts[2], outputs=text_input)
            
            # Batch processing handlers
            def parse_uploaded_file(file_obj):
                if not file_obj:
                    return []
                try:
                    import csv, json, io
                    fname = file_obj.name if hasattr(file_obj, 'name') else ''
                    file_bytes = file_obj.read()
                    text = file_bytes.decode('utf-8') if isinstance(file_bytes, (bytes, bytearray)) else str(file_bytes)
                    
                    if fname.endswith('.json'):
                        data = json.loads(text)
                        if isinstance(data, list):
                            if all(isinstance(x, str) for x in data):
                                return data
                            else:
                                return [str(x.get('text', '')) for x in data if isinstance(x, dict) and 'text' in x]
                    elif fname.endswith('.csv'):
                        rows = []
                        reader = csv.DictReader(io.StringIO(text))
                        for r in reader:
                            if 'text' in r:
                                rows.append(r['text'])
                            else:
                                rows.append(','.join(r.values()))
                        return rows
                    else:
                        return [l for l in text.splitlines() if l.strip()]
                except Exception:
                    return []
            
            def run_batch_background(textarea_value, file_obj, group_a_filter, group_b_filter, save_path_val):
                texts = [l.strip() for l in (textarea_value or '').splitlines() if l.strip()]
                uploaded = parse_uploaded_file(file_obj)
                if uploaded:
                    texts.extend(uploaded)
                
                total = len(texts)
                if total == 0:
                    return {
                        progress_text: "⚠️ No texts provided",
                        progress_bar: 0,
                        job_id_text: "",
                        batch_summary_html: "<div style='padding: 24px; background: linear-gradient(135deg, #fef3c7, #fde68a); border-left: 5px solid #f59e0b; border-radius: 16px;'><strong style='font-size: 16px; color: #92400e;'>⚠️ Warning:</strong> <span style='color: #78350f; margin-left: 8px;'>No texts provided for batch processing.</span></div>",
                        comparison_html: ""
                    }
                
                job_id = f"job_{int(time.time()*1000)}"
                self._jobs[job_id] = {
                    'status': 'queued',
                    'total': total,
                    'processed': 0,
                    'results': [],
                    'summary': None,
                    'created_at': time.time()
                }
                
                import threading
                
                def _worker(jid, texts_local, ga_filter, gb_filter, save_path_local):
                    def progress_cb(processed, total_count):
                        self._jobs[jid]['processed'] = processed
                        self._jobs[jid]['status'] = 'running' if processed < total_count else 'finalizing'
                    
                    try:
                        results = self.analyze_batch(texts_local, progress_callback=progress_cb)
                        self._jobs[jid]['results'] = results
                        self.last_batch_results = results
                        self._jobs[jid]['summary'] = self.summarize_batch(results)
                        
                        group_a = []
                        group_b = []
                        if ga_filter:
                            group_a = [r for r in results if ga_filter.lower() in r.get('text', '').lower()]
                        if gb_filter:
                            group_b = [r for r in results if gb_filter.lower() in r.get('text', '').lower()]
                        if group_a and group_b:
                            self._jobs[jid]['comparison'] = self.compare_groups(group_a, group_b)
                        else:
                            self._jobs[jid]['comparison'] = None
                        
                        if save_path_local:
                            try:
                                if save_path_local.endswith('.json'):
                                    save_results_json(save_path_local, results)
                                elif save_path_local.endswith('.csv'):
                                    save_results_csv(save_path_local, results)
                                self._jobs[jid]['export_path'] = save_path_local
                            except Exception as e:
                                self._jobs[jid]['export_error'] = str(e)
                        
                        self._jobs[jid]['status'] = 'completed'
                    except Exception as e:
                        self._jobs[jid]['status'] = 'failed'
                        self._jobs[jid]['error'] = str(e)
                
                t = threading.Thread(target=_worker, args=(job_id, texts, group_a_filter, group_b_filter, save_path_val), daemon=True)
                t.start()
                
                return {
                    progress_text: "🚀 Job queued successfully",
                    progress_bar: 0,
                    job_id_text: job_id,
                    batch_summary_html: "<div style='padding: 24px; background: linear-gradient(135deg, #d1fae5, #a7f3d0); border-left: 5px solid #10b981; border-radius: 16px;'><strong style='font-size: 16px; color: #065f46;'>✅ Success:</strong> <span style='color: #047857; margin-left: 8px;'>Batch job started. Use 'Refresh Status' to check progress.</span></div>",
                    comparison_html: ""
                }
            
            run_batch_btn.click(
                run_batch_background,
                inputs=[batch_textarea, file_upload, group_a_select, group_b_select, save_path],
                outputs=[progress_text, progress_bar, job_id_text, batch_summary_html, comparison_html]
            )
            
            def poll_job_status(job_id):
                if not job_id:
                    return {progress_text: "⚠️ No job ID", progress_bar: 0, batch_summary_html: "", comparison_html: ""}
                job = self._jobs.get(job_id)
                if not job:
                    return {progress_text: "❌ Job not found", progress_bar: 0, batch_summary_html: "", comparison_html: ""}
                
                pct = round((job.get('processed', 0) / job.get('total', 1)) * 100, 1) if job.get('total') else 0
                summary_html = ""
                if job.get('summary'):
                    s = job['summary']
                    summary_html = f"""
                    <div style='padding: 28px; background: white; border-radius: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 2px solid #e2e8f0; margin-top: 20px;'>
                        <h4 style='margin: 0 0 20px 0; font-size: 20px; font-weight: 800; color: #1e293b;'>📊 Summary Statistics</h4>
                        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;'>
                            <div style='padding: 20px; background: linear-gradient(135deg, #eff6ff, #dbeafe); border-radius: 16px; border: 2px solid #93c5fd;'>
                                <div style='font-size: 13px; color: #1e40af; font-weight: 700; margin-bottom: 8px;'>Total Analyzed</div>
                                <div style='font-size: 36px; color: #1e3a8a; font-weight: 900;'>{s.get('total', 0)}</div>
                            </div>
                            <div style='padding: 20px; background: linear-gradient(135deg, #fef3c7, #fde68a); border-radius: 16px; border: 2px solid #fcd34d;'>
                                <div style='font-size: 13px; color: #92400e; font-weight: 700; margin-bottom: 8px;'>Avg Bias Score</div>
                                <div style='font-size: 36px; color: #78350f; font-weight: 900;'>{s.get('avg_bias_probability', 0.0):.3f}</div>
                            </div>
                        </div>
                        <div style='margin-top: 20px;'>
                            <strong style='color: #334155; font-size: 15px;'>Class Distribution:</strong>
                            <pre style='background: #f8fafc; padding: 16px; border-radius: 12px; margin-top: 12px; overflow-x: auto; border: 2px solid #e2e8f0; color: #1e293b;'>{json.dumps(s.get('class_counts', {}), indent=2)}</pre>
                        </div>
                    </div>
                    """
                
                compare_html = ""
                if job.get('comparison'):
                    c = job['comparison']
                    compare_html = f"""
                    <div style='padding: 28px; background: white; border-radius: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 2px solid #e2e8f0; margin-top: 20px;'>
                        <h4 style='margin: 0 0 20px 0; font-size: 20px; font-weight: 800; color: #1e293b;'>🔄 Group Comparison</h4>
                        <pre style='background: #f8fafc; padding: 20px; border-radius: 12px; overflow-x: auto; border: 2px solid #e2e8f0; color: #1e293b;'>{json.dumps(c, indent=2)}</pre>
                    </div>
                    """
                
                status_text = job.get('status', 'unknown')
                status_emoji = "⏳" if status_text == "running" else "✅" if status_text == "completed" else "❌" if status_text == "failed" else "🔵"
                
                return {
                    progress_text: f"{status_emoji} Status: {status_text}",
                    progress_bar: pct,
                    batch_summary_html: summary_html,
                    comparison_html: compare_html
                }
            
            refresh_status_btn.click(poll_job_status, inputs=[job_id_text], outputs=[progress_text, progress_bar, batch_summary_html, comparison_html])
            
            def export_last_json():
                try:
                    if not self.last_batch_results:
                        return None, "⚠️ No batch results to export"
                    os.makedirs('./export', exist_ok=True)
                    fname = f"./export/batch_results_{int(time.time())}.json"
                    save_results_json(fname, self.last_batch_results)
                    return fname, f"✅ Saved to {fname}"
                except Exception as e:
                    return None, f"❌ Export failed: {e}"
            
            def export_last_csv():
                try:
                    if not self.last_batch_results:
                        return None, "⚠️ No batch results to export"
                    os.makedirs('./export', exist_ok=True)
                    fname = f"./export/batch_results_{int(time.time())}.csv"
                    save_results_csv(fname, self.last_batch_results)
                    return fname, f"✅ Saved to {fname}"
                except Exception as e:
                    return None, f"❌ Export failed: {e}"
            
            export_json_btn.click(export_last_json, outputs=[export_download, progress_text])
            export_csv_btn.click(export_last_csv, outputs=[export_download, progress_text])
        
        return demo
    
    def analyze_batch(self, texts: List[str], progress_callback=None) -> List[Dict]:
        results = []
        total = len(texts)
        try:
            batch_preds = self.analyzer.detector.predict_batch_batched([t for t in texts])
            for i, (t, pred) in enumerate(zip(texts, batch_preds), 1):
                try:
                    res = self.analyzer.analyze_text(t)
                    res['bias_probability'] = pred.get('bias_probability', res.get('bias_probability'))
                    res['bias_class'] = pred.get('classification', res.get('bias_class'))
                    res['confidence'] = pred.get('confidence', res.get('confidence'))
                except Exception:
                    res = {
                        'text': t,
                        'bias_probability': pred.get('bias_probability', 0.0),
                        'bias_class': pred.get('classification', 'UNKNOWN'),
                        'confidence': pred.get('confidence', 0.0),
                        'top_biased_words': [],
                        'shap_scores': [],
                        'counterfactuals': [],
                        'timestamp': time.time()
                    }
                results.append(res)
                if progress_callback:
                    try:
                        progress_callback(i, total)
                    except Exception:
                        pass
        except Exception:
            for i, t in enumerate(texts, 1):
                try:
                    res = self.analyzer.analyze_text(t)
                    results.append(res)
                except Exception as e:
                    results.append({
                        'text': t,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                if progress_callback:
                    try:
                        progress_callback(i, total)
                    except Exception:
                        pass
        return results
    
    def summarize_batch(self, results: List[Dict]) -> Dict[str, Any]:
        total = len(results)
        if total == 0:
            return {}
        bias_probs = [r.get('bias_probability', 0.0) for r in results if 'bias_probability' in r]
        classes = {}
        top_words = {}
        for r in results:
            cls = r.get('bias_class', 'UNKNOWN')
            classes[cls] = classes.get(cls, 0) + 1
            for w in r.get('top_biased_words', []):
                top_words[w] = top_words.get(w, 0) + 1
        avg_bias = sum(bias_probs) / len(bias_probs) if bias_probs else 0.0
        return {
            'total': total,
            'avg_bias_probability': avg_bias,
            'class_counts': classes,
            'top_words_frequency': sorted(top_words.items(), key=lambda x: x[1], reverse=True)[:20]
        }
    
    def compare_groups(self, group_a: List[Dict], group_b: List[Dict]) -> Dict[str, Any]:
        def stats(group):
            probs = [r.get('bias_probability', 0.0) for r in group if 'bias_probability' in r]
            avg = sum(probs) / len(probs) if probs else 0.0
            return {'count': len(group), 'avg_bias': avg}
        a_stats = stats(group_a)
        b_stats = stats(group_b)
        delta = a_stats['avg_bias'] - b_stats['avg_bias']
        return {
            'group_a': a_stats,
            'group_b': b_stats,
            'avg_bias_delta': delta
        }


def main():
    print("🚀 Launching BiasGuard Pro Dashboard (Enhanced UI)...")
    dashboard = BiasGuardDashboard()
    demo = dashboard.create_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()