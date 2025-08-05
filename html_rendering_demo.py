#!/usr/bin/env python3
"""
Demo script showing the new HTML rendering capabilities for state-preserving notebooks.
"""

import pandas as pd
import matplotlib.pyplot as plt
from validmind.vm_models.result.result import TestResult
from validmind.utils import display

def demo_html_rendering():
    """Demonstrate HTML rendering with state preservation."""
    print("🎯 ValidMind HTML Rendering Demo")
    print("=" * 50)
    
    # Create a sample test result
    result = TestResult(
        name="Model Performance Analysis",
        result_id="model_performance_analysis", 
        description="## Analysis Summary\n\nThis analysis shows **excellent model performance** with high accuracy across all metrics. The model demonstrates strong predictive capabilities.",
        metric=0.94,
        passed=True
    )
    
    # Add performance metrics table
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'Training': [0.94, 0.92, 0.91, 0.92, 0.96],
        'Validation': [0.93, 0.91, 0.90, 0.90, 0.95],
        'Test': [0.94, 0.92, 0.89, 0.90, 0.94]
    })
    result.add_table(metrics_df, title="Model Performance Metrics")
    
    # Add feature importance table
    features_df = pd.DataFrame({
        'Feature': ['Income', 'Age', 'Credit_Score', 'Employment_Length', 'Debt_Ratio'],
        'Importance': [0.35, 0.22, 0.18, 0.15, 0.10],
        'P_Value': [0.001, 0.002, 0.005, 0.01, 0.03]
    })
    result.add_table(features_df, title="Feature Importance Analysis")
    
    # Add a performance chart
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    training_scores = [0.94, 0.92, 0.91, 0.92]
    test_scores = [0.94, 0.92, 0.89, 0.90]
    
    x = range(len(metrics))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], training_scores, width, label='Training', alpha=0.8)
    ax.bar([i + width/2 for i in x], test_scores, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Training vs Test')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    result.add_figure(fig)
    
    # Display using the new HTML rendering
    print("\n📊 Displaying result with HTML rendering (state-preserving):")
    display(result)
    
    print("\n✨ Benefits of HTML Rendering:")
    print("• ✅ State preserved when notebook is saved")
    print("• ✅ Works across all Jupyter environments")
    print("• ✅ No dependency on ipywidgets backend")
    print("• ✅ Consistent rendering when shared")
    print("• ✅ Interactive elements with pure HTML/CSS/JS")
    
    # Show the raw HTML length for reference
    html_content = result.to_html()
    print(f"\n📝 Generated HTML size: {len(html_content):,} characters")
    
    print("\n🎉 Demo complete! The result above will retain its state when you save this notebook.")

if __name__ == "__main__":
    demo_html_rendering()