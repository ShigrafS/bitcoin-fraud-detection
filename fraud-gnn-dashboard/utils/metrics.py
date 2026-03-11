import pandas as pd
import plotly.express as px

def model_comparison(metrics_dict):
                                                                        
    if not metrics_dict:
        return None
        
    df_rows = []
    for model, metrics in metrics_dict.items():
        row = {"Model": model}
        row.update(metrics)
        df_rows.append(row)
        
    df = pd.DataFrame(df_rows)
    
                                            
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(df_melted, x="Model", y="Score", color="Metric", barmode="group",
                 title="Model Performance Comparison")
    
    return fig

def radar_chart(metrics_dict):
    if not metrics_dict:
        return None
        
    df_rows = []
    for model, metrics in metrics_dict.items():
        row = {"Model": model}
        row.update(metrics)
        df_rows.append(row)
        
    df = pd.DataFrame(df_rows)
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    fig = px.line_polar(df_melted, r="Score", theta="Metric", color="Model", line_close=True,
                        title="Model Performance Radar")
    return fig

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    import plotly.figure_factory as ff
    
    x = ['Predicted Normal', 'Predicted Fraud']
    y = ['Actual Normal', 'Actual Fraud']
    
                                                
    z = cm[::-1]
    y = y[::-1]
    
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
    fig.update_layout(title=title, margin=dict(t=50, l=100))
    return fig
