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
        
        # Auto-detect model path
        if model_path is None:
            model_path = self._auto_detect_model_path()
        
        print(f"📁 Using model path: {model_path}")
        self.detector = BiasDetector(model_path)
        self.explainer = SHAPExplainer(model_path)
        self.counterfactuals = CounterfactualGenerator()
        print("✅ BiasGuard Pro initialized successfully!\n")
    
    def _auto_detect_model_path(self) -> str:
        """Auto-detect where model files are located"""
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
        """Complete analysis of text for bias"""
        print(f"🔍 Analyzing: '{text}'")
        
        # Get bias prediction
        bias_result = self.detector.predict_bias(text)
        
        # Get SHAP explanations
        shap_results = self.explainer.get_shap_values(text)
        print(f"   Top biased words: {[w for w, s in shap_results[:3]]}")
        
        # Generate counterfactuals
        counterfactuals = self.counterfactuals.generate_counterfactuals(text, shap_results)
        
        return {
            'text': text,
            'bias_probability': bias_result['bias_probability'],
            'bias_class': bias_result['classification'],
            'confidence': bias_result['confidence'],
            'top_biased_words': [w for w, s in shap_results[:3]],
            'shap_scores': shap_results[:10],
            'counterfactuals': counterfactuals,
            'timestamp': time.time()
        }


class BiasGuardDashboard:
    def __init__(self):
        print("🎨 Initializing BiasGuard Pro Dashboard...")
        self.analyzer = BiasGuardPro()
        self.analysis_history = []
        # store last batch run results for exports
        self.last_batch_results: List[Dict] = []
        # simple in-memory job store for background batch jobs
        self._jobs: Dict[str, Dict] = {}
        
        # Sample texts for quick testing
        self.sample_texts = [
            "Women should be nurses because they are compassionate.",
            "Men are naturally better at engineering roles.",
            "The female secretary was very emotional today.",
            "He needs to be more aggressive to succeed in business.",
            "People with technical skills excel in engineering roles."
        ]
        
        print("✅ Dashboard initialized successfully!")
    
    def create_bias_meter(self, bias_prob: float) -> go.Figure:
        """Create a bias probability meter"""
        bias_color = "darkred" if bias_prob > 0.5 else "darkgreen"
        bias_label = "HIGH BIAS" if bias_prob > 0.5 else "LOW BIAS"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = bias_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Bias Probability - {bias_label}", 'font': {'size': 20}},
            delta = {'reference': 0.5, 'increasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': bias_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.3], 'color': '#d4edda'},  # Light green
                    {'range': [0.3, 0.7], 'color': '#fff3cd'},  # Light yellow
                    {'range': [0.7, 1], 'color': '#f8d7da'}  # Light red
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=60, b=20),
            font={'size': 14}
        )
        return fig
    
    def create_shap_chart(self, shap_scores: List) -> go.Figure:
        """Create an enhanced bar chart of SHAP scores"""
        if not shap_scores:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(text="No significant biased words detected",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, xanchor='center', yanchor='middle',
                             showarrow=False, font=dict(size=16))
            return fig
            
        words = [word for word, score in shap_scores[:8]]
        scores = [score for word, score in shap_scores[:8]]
        
        # Color based on bias direction and intensity
        colors = []
        for score in scores:
            if score > 0.1:
                colors.append('#dc3545')  # Strong red for high bias
            elif score > 0:
                colors.append('#ff6b6b')  # Medium red for moderate bias
            elif score < -0.1:
                colors.append('#28a745')  # Strong green for bias-reducing
            else:
                colors.append('#6c757d')  # Gray for neutral
        
        fig = go.Figure(go.Bar(
            x=scores,
            y=words,
            orientation='h',
            marker_color=colors,
            hovertemplate='<b>%{y}</b><br>SHAP Score: %{x:.3f}<br>%{customdata}<extra></extra>',
            customdata=[f"{'Increases' if score > 0 else 'Decreases'} bias detection" for score in scores]
        ))
        
        fig.update_layout(
            title={
                'text': "Word Importance Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="SHAP Score (Positive = Biased)",
            yaxis_title="Words",
            height=350,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        # Add zero line
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
        
        return fig
    
    def create_stereotype_breakdown(self, shap_scores: List) -> go.Figure:
        """Create a pie chart of stereotype categories"""
        if not shap_scores:
            fig = go.Figure()
            fig.add_annotation(text="No stereotype categories identified",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, xanchor='center', yanchor='middle',
                             showarrow=False, font=dict(size=14))
            return fig
        
        # Categorize words into stereotype types
        categories = {
            'Gender Words': 0,
            'Professions': 0,
            'Traits': 0,
            'Comparatives': 0
        }
        
        gender_words = {'women', 'men', 'woman', 'man', 'she', 'he', 'female', 'male'}
        professions = {'nurse', 'engineer', 'secretary', 'teacher', 'doctor', 'ceo'}
        traits = {'emotional', 'compassionate', 'aggressive', 'decisive', 'nurturing'}
        comparatives = {'better', 'best', 'naturally', 'should', 'needs'}
        
        for word, score in shap_scores:
            word_lower = word.lower()
            if word_lower in gender_words:
                categories['Gender Words'] += abs(score)
            elif word_lower in professions:
                categories['Professions'] += abs(score)
            elif word_lower in traits:
                categories['Traits'] += abs(score)
            elif word_lower in comparatives:
                categories['Comparatives'] += abs(score)
        
        # Filter out zero categories
        categories = {k: v for k, v in categories.items() if v > 0}
        
        if not categories:
            fig = go.Figure()
            fig.add_annotation(text="No clear stereotype pattern",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, xanchor='center', yanchor='middle',
                             showarrow=False, font=dict(size=14))
            return fig
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        
        fig = go.Figure(data=[go.Pie(
            labels=list(categories.keys()),
            values=list(categories.values()),
            hole=.3,
            marker_colors=colors[:len(categories)],
            hovertemplate='<b>%{label}</b><br>Impact: %{value:.3f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': "Stereotype Category Breakdown",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        return fig
    
    def highlight_biased_words(self, text: str, shap_scores: List, top_k: int = 5) -> str:
        """Enhanced text highlighting with intensity based on SHAP scores"""
        if not shap_scores:
            return f"<div style='padding: 15px; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa;'>{text}</div>"
        
        # Create word-to-score mapping
        word_scores = {word.lower(): abs(score) for word, score in shap_scores if score > 0}
        
        # Split text and highlight
        words = text.split()
        highlighted_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in word_scores:
                score = word_scores[clean_word]
                # Intensity based on score
                if score > 0.3:
                    color = "#ff4444"  # High bias - bright red
                elif score > 0.1:
                    color = "#ff8888"  # Medium bias - medium red
                else:
                    color = "#ffcccc"  # Low bias - light red
                
                highlighted_words.append(
                    f"<mark style='background-color: {color}; padding: 2px 4px; border-radius: 4px; border: 1px solid #ff0000;'><b>{word}</b></mark>"
                )
            else:
                highlighted_words.append(word)
        
        highlighted_text = " ".join(highlighted_words)
        return f"""
        <div style='
            padding: 15px; 
            border: 2px solid #e9ecef; 
            border-radius: 8px; 
            background: white;
            font-size: 16px;
            line-height: 1.6;
        '>
            {highlighted_text}
        </div>
        """
    
    def analyze_text_for_dashboard(self, text: str, show_detailed: bool = True) -> Dict[str, Any]:
        """Enhanced analysis with multi-modal explanations"""
        if not text.strip():
            return {
                "error": "❌ Please enter some text to analyze.",
                "bias_probability": 0.0,
                "bias_class": "NEUTRAL"
            }
        
        try:
            # Run analysis
            result = self.analyzer.analyze_text(text)
            
            # Store in history
            self.analysis_history.append(result)
            if len(self.analysis_history) > 10:
                self.analysis_history.pop(0)
            
            # Create all visualizations
            bias_meter = self.create_bias_meter(result['bias_probability'])
            shap_chart = self.create_shap_chart(result['shap_scores'])
            stereotype_chart = self.create_stereotype_breakdown(result['shap_scores'])
            highlighted_text = self.highlight_biased_words(text, result['shap_scores'])
            
            # Enhanced counterfactuals display
            counterfactuals_html = "<div style='margin: 15px 0;'>"
            if result['counterfactuals']:
                for i, cf in enumerate(result['counterfactuals'], 1):
                    counterfactuals_html += f"""
                    <div style='
                        margin: 12px 0; 
                        padding: 15px; 
                        background: linear-gradient(135deg, #e8f5e8, #d4edda); 
                        border: 1px solid #c3e6cb;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    '>
                        <div style='font-size: 14px; color: #155724; margin-bottom: 5px;'>
                            <strong>🔄 Suggestion {i}</strong>
                        </div>
                        <div style='font-size: 16px; color: #0c5460;'>{cf}</div>
                    </div>
                    """
            else:
                counterfactuals_html += """
                <div style='
                    padding: 20px; 
                    background: #f8f9fa; 
                    border-radius: 8px; 
                    text-align: center;
                    color: #6c757d;
                    font-style: italic;
                '>
                    No counterfactual suggestions available for this text.
                </div>
                """
            counterfactuals_html += "</div>"
            
            # Create explanation summary
            explanation_summary = self._create_explanation_summary(result)
            
            return {
                "bias_probability": result['bias_probability'],
                "bias_class": result['bias_class'],
                "bias_meter": bias_meter,
                "shap_chart": shap_chart,
                "stereotype_chart": stereotype_chart,
                "highlighted_text": highlighted_text,
                "counterfactuals_html": counterfactuals_html,
                "explanation_summary": explanation_summary,
                "top_words": result['top_biased_words'],
                "confidence": result['confidence'],
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"❌ Analysis failed: {str(e)}",
                "bias_probability": 0.0,
                "bias_class": "ERROR"
            }
    
    def _create_explanation_summary(self, result: Dict) -> str:
        """Create a natural language summary of the analysis"""
        bias_level = "highly biased" if result['bias_probability'] > 0.7 else \
                   "moderately biased" if result['bias_probability'] > 0.5 else "relatively neutral"
        
        top_words = result['top_biased_words']
        if top_words:
            word_summary = f"The analysis detected potential bias in words like '{', '.join(top_words[:3])}'"
        else:
            word_summary = "No strongly biased words were identified"
        
        summary = f"""
        <div style='
            padding: 15px; 
            background: #e3f2fd; 
            border-radius: 8px; 
            border-left: 4px solid #2196F3;
            margin: 10px 0;
        '>
            <h4 style='margin: 0 0 10px 0; color: #0d47a1;'>📋 Analysis Summary</h4>
            <p style='margin: 5px 0;'>This text is <strong>{bias_level}</strong> with a probability of <strong>{result['bias_probability']:.3f}</strong>.</p>
            <p style='margin: 5px 0;'>{word_summary}.</p>
            <p style='margin: 5px 0;'>The model is <strong>{result['confidence']:.1%} confident</strong> in this assessment.</p>
        </div>
        """
        return summary

    # ------------------ Batch processing utilities ------------------
    def analyze_batch(self, texts: List[str], progress_callback=None) -> List[Dict]:
        """Analyze a list of texts and optionally report progress via callback.

        progress_callback should accept two args: (processed_count, total_count)
        """
        results = []
        total = len(texts)
        # Prefer batched model inference where available
        try:
            # Use batched detector outputs (only probabilities); fall back to per-text analyzer for full outputs
            batch_preds = self.analyzer.detector.predict_batch_batched([t for t in texts])
            for i, (t, pred) in enumerate(zip(texts, batch_preds), 1):
                # enrich with explanations via analyzer.analyze_text but only if needed
                try:
                    res = self.analyzer.analyze_text(t)
                    # override the bias_probability with faster batched output to keep consistency
                    res['bias_probability'] = pred.get('bias_probability', res.get('bias_probability'))
                    res['bias_class'] = pred.get('classification', res.get('bias_class'))
                    res['confidence'] = pred.get('confidence', res.get('confidence'))
                except Exception:
                    # If analyzer fails, fall back to aggregated pred
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
            # Fallback to slow per-item processing
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
        """Produce summary statistics over batch results."""
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
        """Compare two groups of batch results and return simple comparative metrics."""
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
    
    def create_dashboard(self):
        """Create the enhanced Gradio dashboard with PROGRESSIVE DISCLOSURE"""
        
        # Enhanced CSS
        custom_css = """
        .bias-guard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .success-box {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .warning-box {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .bias-high {
            color: #dc3545;
            font-weight: bold;
            background: #f8d7da;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .bias-low {
            color: #28a745;
            font-weight: bold;
            background: #d4edda;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .tab-buttons {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .explanation-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
        }
        """
        
        with gr.Blocks(css=custom_css, title="BiasGuard Pro - Gender Bias Detector") as demo:
            
            # Header
            gr.Markdown("""
            <div style='text-align: center; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
                <h1 style='margin: 0; font-size: 2.5em;'>🛡️ BiasGuard Pro</h1>
                <h3 style='margin: 10px 0 0 0; font-weight: 300;'>Auditing and Mitigating Gendered Stereotypes in Career Recommendations</h3>
            </div>
            """)
            
            # Main content
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    with gr.Group():
                        gr.Markdown("### 📝 Input Text")
                        text_input = gr.Textbox(
                            label="Enter career recommendation text to analyze",
                            placeholder="e.g., 'Women should be nurses because they are compassionate'",
                            lines=4,
                            max_lines=6,
                            show_label=False
                        )
                    
                    # Sample texts
                    with gr.Group():
                        gr.Markdown("### 💡 Quick Examples")
                        with gr.Row():
                            for i, sample in enumerate(self.sample_texts[:3]):
                                gr.Button(
                                    sample[:40] + "...", 
                                    size="sm", 
                                    variant="secondary"
                                ).click(lambda x=sample: x, outputs=text_input)
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze Text for Bias", 
                        variant="primary", 
                        size="lg",
                        elem_id="analyze-btn"
                    )
                
                with gr.Column(scale=1):
                    # Info panel
                    with gr.Group():
                        gr.Markdown("### ℹ️ How It Works")
                        gr.Markdown("""
                        **1. Detection**: AI model identifies gendered stereotypes  
                        **2. Explanation**: SHAP analysis highlights biased words  
                        **3. Mitigation**: Suggests neutral alternatives  
                        
                        *Built with DistilBERT + SHAP + Advanced NLP*
                        """)
            
            # Results section - PROGRESSIVE DISCLOSURE
            with gr.Column(visible=False) as results_section:
                gr.Markdown("## 📊 Analysis Results")
                
                # LEVEL 1: Quick overview (immediately visible)
                with gr.Row():
                    with gr.Column(scale=1):
                        bias_meter = gr.Plot(label="Bias Probability Meter")
                    
                    with gr.Column(scale=2):
                        explanation_summary = gr.HTML(label="Analysis Summary")
                
                # LEVEL 2: Multi-modal explanation tabs (user chooses depth)
                gr.Markdown("### 🔍 Detailed Explanations")
                with gr.Tabs() as explanation_tabs:
                    # TAB 1: Word Analysis
                    with gr.TabItem("📊 Word Analysis"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                shap_chart = gr.Plot(label="Word Importance")
                            with gr.Column(scale=1):
                                stereotype_chart = gr.Plot(label="Stereotype Patterns")
                                top_words_display = gr.Textbox(
                                    label="Top Biased Words", 
                                    interactive=False,
                                    max_lines=2
                                )
                    
                    # TAB 2: Text View
                    with gr.TabItem("📝 Text View"):
                        highlighted_text = gr.HTML(label="Original Text with Highlights")
                    
                    # TAB 3: Alternatives
                    with gr.TabItem("🔄 Alternatives"):
                        gr.Markdown("### Neutral Alternative Suggestions")
                        counterfactuals_display = gr.HTML(label="Counterfactual Recommendations")

                    # TAB 4: Batch & Compare
                    with gr.TabItem("📚 Batch & Compare"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### 🧪 Batch Input")
                                batch_textarea = gr.Textbox(
                                    label="Paste multiple texts (one per line)",
                                    lines=8,
                                    placeholder="Enter one text per line or upload a file",
                                    show_label=False
                                )
                                file_upload = gr.File(label="Upload .txt/.csv/.json file (optional)")
                                group_a_select = gr.Textbox(label="Group A filter (optional, substring)", placeholder="e.g., 'female' - texts containing this go to group A")
                                group_b_select = gr.Textbox(label="Group B filter (optional, substring)", placeholder="e.g., 'male' - texts containing this go to group B")
                                run_batch_btn = gr.Button("🔁 Start Background Batch", variant="primary")
                                refresh_status_btn = gr.Button("🔄 Refresh Job Status")
                            with gr.Column(scale=1):
                                gr.Markdown("### ⏳ Progress")
                                progress_text = gr.Textbox(interactive=False, label="Progress")
                                progress_bar = gr.Number(value=0, label="Percent Complete", precision=1)
                                job_id_text = gr.Textbox(interactive=False, label="Job ID")
                                gr.Markdown("### 📦 Export")
                                export_json_btn = gr.Button("Export JSON")
                                export_csv_btn = gr.Button("Export CSV")
                                save_path = gr.Textbox(label="Save path (optional)", placeholder="e.g., ./export/results.json")
                                export_download = gr.File(label="Download Export", interactive=False)
                        gr.Markdown("### 🧾 Batch Summary")
                        batch_summary_html = gr.HTML()
                        comparison_html = gr.HTML()

                # ... existing event handlers remain ...

                # Batch handlers
                def parse_uploaded_file(file_obj):
                    if not file_obj:
                        return []
                    try:
                        import os, io, csv, json
                        fname = file_obj.name if hasattr(file_obj, 'name') else ''
                        file_bytes = file_obj.read()
                        text = file_bytes.decode('utf-8') if isinstance(file_bytes, (bytes, bytearray)) else str(file_bytes)
                        if fname.endswith('.json'):
                            data = json.loads(text)
                            # Support either list of strings or list of objects with 'text'
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
                            # treat as plain text with lines
                            return [l for l in text.splitlines() if l.strip()]
                    except Exception:
                        return []

                def run_batch_background(textarea_value, file_obj, group_a_filter, group_b_filter, save_path_val):
                    # Build list of texts
                    texts = [l.strip() for l in (textarea_value or '').splitlines() if l.strip()]
                    uploaded = parse_uploaded_file(file_obj)
                    if uploaded:
                        texts.extend(uploaded)

                    total = len(texts)
                    if total == 0:
                        return {progress_text: "No texts provided.", progress_bar: 0, job_id_text: "", batch_summary_html: "<div class='warning-box'>No texts provided for batch.</div>", comparison_html: ""}

                    # Create job id
                    job_id = f"job_{int(time.time()*1000)}"
                    self._jobs[job_id] = {
                        'status': 'queued',
                        'total': total,
                        'processed': 0,
                        'results': [],
                        'summary': None,
                        'created_at': time.time()
                    }

                    # Launch background thread to process the batch
                    import threading

                    def _worker(jid, texts_local, ga_filter, gb_filter, save_path_local):
                        def progress_cb(processed, total_count):
                            self._jobs[jid]['processed'] = processed
                            self._jobs[jid]['status'] = 'running' if processed < total_count else 'finalizing'

                        try:
                            results = self.analyze_batch(texts_local, progress_callback=progress_cb)
                            self._jobs[jid]['results'] = results
                            self._jobs[jid]['summary'] = self.summarize_batch(results)
                            # Grouping
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

                            # Save if requested
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

                    return {progress_text: "Job queued", progress_bar: 0, job_id_text: job_id, batch_summary_html: "<div class='success-box'>Job started in background. Use Refresh Job Status.</div>", comparison_html: ""}

                run_batch_btn.click(
                    run_batch_background,
                    inputs=[batch_textarea, file_upload, group_a_select, group_b_select, save_path],
                    outputs=[progress_text, progress_bar, job_id_text, batch_summary_html, comparison_html]
                )

                def poll_job_status(job_id):
                    if not job_id:
                        return {progress_text: "No job id provided.", progress_bar: 0, batch_summary_html: "", comparison_html: ""}
                    job = self._jobs.get(job_id)
                    if not job:
                        return {progress_text: "Job not found.", progress_bar: 0, batch_summary_html: "", comparison_html: ""}

                    pct = round((job.get('processed', 0) / job.get('total', 1)) * 100, 1) if job.get('total') else 0
                    summary_html = ""
                    if job.get('summary'):
                        s = job['summary']
                        summary_html = f"<div class='success-box'><strong>Total:</strong> {s.get('total', 0)} &nbsp; <strong>Avg bias:</strong> {s.get('avg_bias_probability', 0.0):.3f}</div>"
                        summary_html += "<div style='margin:8px 0;'><strong>Class counts:</strong><pre>" + json.dumps(s.get('class_counts', {}), indent=2) + "</pre></div>"
                        summary_html += "<div style='margin:8px 0;'><strong>Top words:</strong><pre>" + json.dumps(s.get('top_words_frequency', []), indent=2) + "</pre></div>"

                    compare_html = ""
                    if job.get('comparison'):
                        compare_html = f"<div class='explanation-section'><strong>Group comparison:</strong><pre>{json.dumps(job['comparison'], indent=2)}</pre></div>"

                    status_text = job.get('status', 'unknown')
                    return {progress_text: f"Status: {status_text}", progress_bar: pct, batch_summary_html: summary_html, comparison_html: compare_html}

                refresh_status_btn.click(poll_job_status, inputs=[job_id_text], outputs=[progress_text, progress_bar, batch_summary_html, comparison_html])

                def export_last_json():
                    # Export last batch results to ./export and return file path
                    try:
                        if not self.last_batch_results:
                            return None, "No batch results to export. Run a batch first."
                        os.makedirs('./export', exist_ok=True)
                        fname = f"./export/batch_results_{int(time.time())}.json"
                        save_results_json(fname, self.last_batch_results)
                        return fname, f"Saved JSON to {fname}"
                    except Exception as e:
                        return None, f"Failed to export JSON: {e}"

                def export_last_csv():
                    try:
                        if not self.last_batch_results:
                            return None, "No batch results to export. Run a batch first."
                        os.makedirs('./export', exist_ok=True)
                        fname = f"./export/batch_results_{int(time.time())}.csv"
                        save_results_csv(fname, self.last_batch_results)
                        return fname, f"Saved CSV to {fname}"
                    except Exception as e:
                        return None, f"Failed to export CSV: {e}"

                export_json_btn.click(export_last_json, inputs=None, outputs=[export_download, progress_text])
                export_csv_btn.click(export_last_csv, inputs=None, outputs=[export_download, progress_text])
                
                # LEVEL 3: Detailed metrics (collapsible, for experts)
                with gr.Accordion("📈 Detailed Metrics", open=False):
                    with gr.Row():
                        with gr.Column():
                            bias_prob_display = gr.Textbox(
                                label="Bias Probability", 
                                interactive=False
                            )
                            bias_class_display = gr.Textbox(
                                label="Classification", 
                                interactive=False
                            )
                        with gr.Column():
                            confidence_display = gr.Textbox(
                                label="Model Confidence", 
                                interactive=False
                            )
            
            # Error display
            error_display = gr.HTML(visible=False)
            
            # Analysis history
            with gr.Accordion("📋 Analysis History (Last 10)", open=False):
                history_html = gr.HTML()
            
            # Footer
            gr.Markdown("""
            ---
            <div style='text-align: center; color: #6c757d;'>
                <strong>BiasGuard Pro v1.0</strong> | Built with ❤️ for fair AI | 
                <a href="https://arxiv.org" target="_blank">Research Paper</a> |
                <a href="https://github.com" target="_blank">GitHub</a>
            </div>
            """)
            
            # Event handlers
            def update_display(text):
                """Enhanced display update with progressive disclosure"""
                if not text.strip():
                    return {
                        results_section: gr.update(visible=False),
                        error_display: gr.update(
                            value="<div class='warning-box'>Please enter some text to analyze.</div>", 
                            visible=True
                        )
                    }
                
                result = self.analyze_text_for_dashboard(text)
                
                if "error" in result:
                    return {
                        results_section: gr.update(visible=False),
                        error_display: gr.update(
                            value=f"<div class='warning-box'>{result['error']}</div>", 
                            visible=True
                        )
                    }
                
                # Update history display
                history_html_value = "<div style='max-height: 300px; overflow-y: auto;'>"
                for analysis in reversed(self.analysis_history):
                    bias_class = "bias-high" if analysis['bias_class'] == 'BIASED' else "bias-low"
                    history_html_value += f"""
                    <div style='
                        margin: 8px 0; 
                        padding: 12px; 
                        border-left: 4px solid #4CAF50; 
                        background: #f8f9fa; 
                        border-radius: 4px;
                    '>
                        <div style='font-weight: bold; margin-bottom: 5px;'>{analysis['text'][:80]}...</div>
                        <div>
                            Probability: <span class='{bias_class}'>{analysis['bias_probability']:.3f}</span> | 
                            Classification: {analysis['bias_class']} |
                            Confidence: {analysis['confidence']:.1%}
                        </div>
                    </div>
                    """
                history_html_value += "</div>"
                
                return {
                    results_section: gr.update(visible=True),
                    error_display: gr.update(visible=False),
                    bias_meter: result['bias_meter'],
                    shap_chart: result['shap_chart'],
                    stereotype_chart: result['stereotype_chart'],
                    highlighted_text: result['highlighted_text'],
                    counterfactuals_display: result['counterfactuals_html'],
                    explanation_summary: result['explanation_summary'],
                    bias_prob_display: f"{result['bias_probability']:.3f}",
                    bias_class_display: result['bias_class'],
                    confidence_display: f"{result['confidence']:.2f}",
                    top_words_display: ", ".join(result['top_words']),
                    history_html: history_html_value
                }
            
            # Connect events
            analyze_btn.click(
                update_display,
                inputs=[text_input],
                outputs=[
                    results_section, error_display, bias_meter, shap_chart,
                    stereotype_chart, highlighted_text, counterfactuals_display,
                    explanation_summary, bias_prob_display, bias_class_display, 
                    confidence_display, top_words_display, history_html
                ]
            )
            
            # Also analyze on Enter key
            text_input.submit(
                update_display,
                inputs=[text_input],
                outputs=[
                    results_section, error_display, bias_meter, shap_chart,
                    stereotype_chart, highlighted_text, counterfactuals_display,
                    explanation_summary, bias_prob_display, bias_class_display,
                    confidence_display, top_words_display, history_html
                ]
            )
        
        return demo


def main():
    """Main function to launch the enhanced dashboard"""
    print("🚀 Launching BiasGuard Pro Dashboard (Day 22 - Progressive Disclosure)...")
    
    # Create and launch dashboard
    dashboard = BiasGuardDashboard()
    demo = dashboard.create_dashboard()
    
    # Launch with public sharing
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        debug=True,
        show_error=True
    )


if __name__ == "__main__":
    main()