import matplotlib.pyplot as plt
import numpy as np

# 1. Learning Curves Comparison
def plot_learning_curves_comparison(baseline_rewards, hybrid_rewards, baseline_mean, hybrid_mean):
    plt.subplot(2, 3, 1)
    x_baseline = range(len(baseline_rewards))
    x_hybrid = range(len(hybrid_rewards))
    plt.plot(x_baseline, baseline_rewards, 'o-', label='Baseline MARL', alpha=0.7)
    plt.plot(x_hybrid, hybrid_rewards, 's-', label='Hybrid Approach', alpha=0.7)
    # Add confidence intervals if we have multiple runs
    if len(baseline_rewards) > 1:
        plt.axhline(y=baseline_mean, color='blue', linestyle='--', alpha=0.5)
        plt.axhline(y=hybrid_mean, color='orange', linestyle='--', alpha=0.5)
    plt.xlabel('Run Number')
    plt.ylabel('Best Episode Reward')
    plt.title('Performance Across Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 2. Performance Distribution (Box Plots)
def plot_performance_distribution_comparison(baseline_rewards, hybrid_rewards):
    """Compare performance distribution between baseline and hybrid approaches using box plots"""       
    plt.subplot(2, 3, 2)
    data = [baseline_rewards, hybrid_rewards]
    box_plot = plt.boxplot(data, labels=['Baseline\nMARL', 'Hybrid\nApproach'], patch_artist=True)
    # Color the boxes
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel('Final Performance')
    plt.title('Performance Distribution')
    plt.grid(True, alpha=0.3)

# 3. Convergence Speed Analysis
def plot_convergence_speed_comparison(baseline_convergence, hybrid_convergence, baseline_rewards, hybrid_rewards):
    """Compare convergence speed between baseline and hybrid approaches"""
    
    plt.subplot(2, 3, 3)
    plt.scatter(baseline_convergence, baseline_rewards, 
            alpha=0.6, label='Baseline', s=60, c='blue')
    plt.scatter(hybrid_convergence, hybrid_rewards, 
            alpha=0.6, label='Hybrid', s=60, c='red')
    plt.xlabel('Iterations to Convergence')
    plt.ylabel('Final Reward')
    plt.title('Convergence Speed vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

##Statistical Significance Visualization
def plot_paired_comparison(baseline_rewards, hybrid_rewards):       
    plt.subplot(2, 2, 4)
        # Paired comparison plot
    for i in range(len(baseline_rewards)):
        plt.plot([1, 2], [baseline_rewards[i], hybrid_rewards[i]], 
                'o-', alpha=0.5, color='gray')
    plt.plot([1, 2], [np.mean(baseline_rewards), np.mean(hybrid_rewards)], 
                'o-', linewidth=3, markersize=10, color='red')
    plt.xticks([1, 2], ['Baseline', 'Hybrid'])
    plt.ylabel('Performance')
    plt.title('Paired Comparison (Each Line = One Run)')

 # 5. Improvement Histogram
def plot_improvement_histogram(baseline_rewards, hybrid_rewards):    
    plt.subplot(2, 3, 5)
    improvements = [(h-b)/b*100 for h, b in zip(hybrid_rewards, baseline_rewards)]
    plt.hist(improvements, bins=min(10, len(improvements)), alpha=0.7, 
                edgecolor='black', color='green')
    plt.xlabel('Improvement (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Improvements')
    plt.axvline(0, color='red', linestyle='--', label='No Improvement', linewidth=2)
    plt.legend()
    plt.grid(True, alpha=0.3)

# 6. Summary Statistics
def generate_comparison_plots(baseline_rewards, hybrid_rewards, baseline_mean, hybrid_mean, baseline_std=None, hybrid_std=None):
    
    plt.subplot(2, 3, 6)
    categories = ['Mean\nReward', 'Max\nReward', 'Min\nReward', 'Std\nReward']
    baseline_stats = [baseline_mean, np.max(baseline_rewards), 
                        np.min(baseline_rewards), baseline_std]
    hybrid_stats = [hybrid_mean, np.max(hybrid_rewards), 
                    np.min(hybrid_rewards), hybrid_std]
        
    x_pos = np.arange(len(categories))
    width = 0.35
        
    plt.bar(x_pos - width/2, baseline_stats, width, label='Baseline', 
                color='lightblue', alpha=0.8)
    plt.bar(x_pos + width/2, hybrid_stats, width, label='Hybrid', 
                color='lightcoral', alpha=0.8)
        
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Statistical Summary')
    plt.xticks(x_pos, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()

#### Additional Plots for Training Analysis ####

def plot_training_efficiency(metrics_by_phase):
    # 1. Training Efficiency Plot
    plt.subplot(2, 2, 1)
    for phase, data in metrics_by_phase.items():
        # Convert timestamps to relative minutes
        start_time = data["timestamps"][0]
        times_minutes = [(t - start_time).total_seconds() / 60.0 for t in data["timestamps"]]
        cumulative_rewards = np.cumsum(data["reward"])
    plt.plot(times_minutes, cumulative_rewards, label=f'{phase.capitalize()}', alpha=0.7)
    plt.xlabel('Training Time (minutes)')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Efficiency')
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_policy_loss_convergence(metrics_by_phase):
    # 2. Policy Loss Convergence
    plt.subplot(2, 2, 2)
    for phase, data in metrics_by_phase.items():
        plt.plot(data["iterations"], data["policy_loss"], label=f'{phase.capitalize()} Policy Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Policy Loss')
    plt.title('Policy Learning Stability')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_value_function_loss_convergence(metrics_by_phase):
    # 3. Value Function Loss (optional extra)
    plt.subplot(2, 2, 3)
    for phase, data in metrics_by_phase.items():
        plt.plot(data["iterations"], data["vf_loss"], label=f'{phase.capitalize()} VF Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Value Function Loss')
    plt.title('VF Loss Convergence')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

