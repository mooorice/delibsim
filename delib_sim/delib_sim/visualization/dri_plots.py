#!/usr/bin/env python
"""DRI Scatter Plot Generation Module."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import Optional
from scipy.stats import gaussian_kde


def create_dri_scatter_plot(ic_df: pd.DataFrame, 
                           group_score: float, 
                           output_path: str, 
                           stage_label: str) -> None:
    """
    Create a DRI scatter plot showing pairwise consistency between participants.
    
    Args:
        ic_df: DataFrame with columns ['P_P', 'P1', 'P2', 'Q_corr', 'R_corr']
        group_score: Group DRI score to display on plot
        output_path: Path where to save the PNG file
        stage_label: Label for the stage (e.g., "Pre-deliberation", "Post-deliberation")
    """
    if ic_df.empty:
        print(f"Warning: Empty IC dataframe for {stage_label}, skipping plot generation.")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract data for plotting
    x_data = ic_df['Q_corr']  # Considerations consistency (x-axis)
    y_data = ic_df['R_corr']  # Preferences consistency (y-axis)
    
    # Calculate density for contour plot
    if len(x_data) > 1:  # Need at least 2 points for density estimation
        try:
            # Create a grid for density estimation
            x_min, x_max = -1, 1
            y_min, y_max = -1, 1
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Calculate kernel density estimation
            values = np.vstack([x_data, y_data])
            kernel = gaussian_kde(values)
            density = np.reshape(kernel(positions).T, xx.shape)
            
            # Find density levels (center, middle, outer)
            max_density = np.max(density)
            level_1 = max_density * 0.7  # Inner level (70% of max density)
            level_2 = max_density * 0.4  # Outer level (40% of max density)
            
            # Plot density contours
            contour = ax.contour(xx, yy, density, levels=[level_2, level_1], 
                               colors=['lightblue', 'blue'], alpha=0.6, linewidths=2)
            ax.contourf(xx, yy, density, levels=[level_2, level_1], 
                       colors=['lightblue', 'blue'], alpha=0.2)
            
            # Add contour labels
            ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')
            
        except Exception as e:
            print(f"Warning: Could not calculate density contours: {e}")
    
    # Create scatter plot with semi-transparent markers
    ax.scatter(x_data, y_data, alpha=0.7, s=50, c='steelblue', edgecolors='darkblue', linewidth=0.5)
    
    # Set axis limits and labels
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Consistency in Considerations', fontsize=12)
    ax.set_ylabel('Consistency in Preferences', fontsize=12)
    
    # Add title with stage information
    ax.set_title(f'DRI Scatter Plot - {stage_label}\nGroup DRI Score: {group_score:.3f}', 
                 fontsize=14, fontweight='bold')
    
    # Add diagonal reference line for perfect consistency
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.6, linewidth=1.5)
    ax.text(-0.35, -0.35, 'Perfect Intersubjective Consistency',
        rotation=45, ha='left', va='bottom')
    
    # Add subtle grid lines to indicate quadrants
    ax.axhline(y=0, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Add quadrant labels
    ax.text(0.5, 0.5, 'High Consideration\nHigh Preference', 
            ha='center', va='center', fontsize=10, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
    ax.text(-0.5, 0.5, 'Low Consideration\nHigh Preference', 
            ha='center', va='center', fontsize=10, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.3))
    ax.text(0.5, -0.5, 'High Consideration\nLow Preference', 
            ha='center', va='center', fontsize=10, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.3))
    ax.text(-0.5, -0.5, 'Low Consideration\nLow Preference', 
            ha='center', va='center', fontsize=10, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', alpha=0.3, label='Density Level 1 (40%)'),
        Patch(facecolor='blue', alpha=0.3, label='Density Level 2 (70%)'),
        ax.scatter([], [], c='steelblue', s=50, label='Participant Pairs', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Saved DRI scatter plot: {output_path}")


def generate_dri_plots_for_iteration(iteration_result: dict, 
                                    output_dir: str, 
                                    timestamp: str) -> None:
    """
    Generate both pre and post-deliberation DRI plots for a single iteration.
    
    Args:
        iteration_result: Result dictionary from a single iteration
        output_dir: Directory where to save the plots
        timestamp: Timestamp string for filename
    """
    iteration_idx = iteration_result.get('iteration', 0)
    
    # Generate pre-deliberation plot
    pre_data = iteration_result.get('pre_deliberation', {})
    pre_ic_data = pre_data.get('ic_dataframe')
    pre_group_score = pre_data.get('group_score', 0.0)
    
    if pre_ic_data is not None:
        # Convert from dict records back to DataFrame if needed
        if isinstance(pre_ic_data, list):
            pre_ic_df = pd.DataFrame(pre_ic_data)
        else:
            pre_ic_df = pre_ic_data
        
        if not pre_ic_df.empty:
            pre_plot_path = os.path.join(output_dir, f"dri_plot_pre_{iteration_idx}_{timestamp}.png")
            create_dri_scatter_plot(pre_ic_df, pre_group_score, pre_plot_path, "Pre-deliberation")
    
    # Generate post-deliberation plot
    post_data = iteration_result.get('post_deliberation', {})
    post_ic_data = post_data.get('ic_dataframe')
    post_group_score = post_data.get('group_score', 0.0)
    
    if post_ic_data is not None:
        # Convert from dict records back to DataFrame if needed
        if isinstance(post_ic_data, list):
            post_ic_df = pd.DataFrame(post_ic_data)
        else:
            post_ic_df = post_ic_data
        
        if not post_ic_df.empty:
            post_plot_path = os.path.join(output_dir, f"dri_plot_post_{iteration_idx}_{timestamp}.png")
            create_dri_scatter_plot(post_ic_df, post_group_score, post_plot_path, "Post-deliberation")
