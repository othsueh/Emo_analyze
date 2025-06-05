import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.cluster import DBSCAN

def drop_rate_calculator(corpus):
    """
    Count the occurrences of each dominant emotion in the corpus.
    
    Args:
        corpus: DataFrame containing a 'dominant_emotion' column
    """
    splits = ['Train', 'Development', 'Test']
    total_ambiguous = 0

    for s in splits:
        split_set = corpus[corpus['Split_Set'] == s]         
        emotion_counts = Counter(split_set['dominant_emotion'])
        ambiguous_count = emotion_counts['ambiguous']
        total_ambiguous += ambiguous_count
        print(f'drop rate of {s} set : {ambiguous_count / len(split_set) * 100:.2f}%')
        
    print(f'Overall drop rate : {total_ambiguous / len(corpus) * 100:.2f}%')


def Normalize_dimensional_emotion(corpus):
    act_min = corpus['EmoAct'].min()
    act_max = corpus['EmoAct'].max()
    val_min = corpus['EmoVal'].min()
    val_max = corpus['EmoVal'].max()
    print(f'Activation: Min {act_min} Max{act_max}')
    print(f'Valence: Min {val_min} Max{val_max}')

    # Normalize EmoAct to [0,1]
    corpus['EmoAct'] = (corpus['EmoAct'] - act_min) / (act_max - act_min)
    # Normalize EmoVal to [0,1]
    corpus['EmoVal'] = (corpus['EmoVal'] - val_min) / (val_max - val_min)

    corpus = corpus.drop(columns='EmoDom')

    return corpus


def get_main_plot(emotion_cols, df):
    # Create a single figure with appropriate size
    fig, ax = plt.subplots(figsize=(8, 5))
    #! larger graph can help
    
    # Plot all emotions in the plot
    for emotion in emotion_cols:
        mask = df['dominant_emotion'] == emotion
        ax.scatter(df.loc[mask, 'EmoVal'], 
                    df.loc[mask, 'EmoAct'],
                    label=emotion,
                    alpha=0.7)
    
    # Add circular grid
    center = (0.5, 0.5)
    radius = 0.5
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Draw concentric circles
    for r in np.linspace(0, radius, 6):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        ax.plot(x, y, 'gray', alpha=0.3)
    
    # Draw radial lines
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x = [center[0], center[0] + radius * np.cos(angle)]
        y = [center[1], center[1] + radius * np.sin(angle)]
        ax.plot(x, y, 'gray', alpha=0.3)
    
    # Set titles and labels
    ax.set_title('Emotion Distribution')
    ax.set_xlabel('Emotional Valence (EmoVal)')
    ax.set_ylabel('Emotional Activation (EmoAct)')
    ax.legend()
    ax.axis('equal')
    ax.set_xlim(center[0] - radius - 0.1, center[0] + radius + 0.1)
    ax.set_ylim(center[1] - radius - 0.1, center[1] + radius + 0.1)
    
    plt.tight_layout()
    plt.show()

def get_full_plot(emotion_cols,df):
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # Calculate grid dimensions
    n_emotions = len(emotion_cols)
    n_cols = 3  # We'll use 3 columns
    n_rows = 1 + ((n_emotions - 1) // n_cols) + 1  # First row for main plot + rows needed for emotions

    # Main plot (taking up 3 spaces in the first row)
    ax_main = plt.subplot2grid((n_rows, n_cols), (0, 1), colspan=1)

    # Plot all emotions in the main plot
    for emotion in emotion_cols:
        mask = df['dominant_emotion'] == emotion
        ax_main.scatter(df.loc[mask, 'EmoVal'], 
                    df.loc[mask, 'EmoAct'],
                    label=emotion,
                    alpha=0.7)

    # Add circular grid to main plot
    center = (0.5, 0.5)  # Changed center to 0.5, 0.5
    radius = 0.5  # Changed radius to 0.5
    theta = np.linspace(0, 2*np.pi, 100)

    # Draw concentric circles in main plot
    for r in np.linspace(0, radius, 6):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        ax_main.plot(x, y, 'gray', alpha=0.3)

    # Draw radial lines in main plot
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x = [center[0], center[0] + radius * np.cos(angle)]
        y = [center[1], center[1] + radius * np.sin(angle)]
        ax_main.plot(x, y, 'gray', alpha=0.3)

    ax_main.set_title('All Emotions Distribution')
    ax_main.set_xlabel('Emotional Valence (EmoVal)')
    ax_main.set_ylabel('Emotional Activation (EmoAct)')
    ax_main.legend()
    ax_main.axis('equal')
    ax_main.set_xlim(center[0] - radius - 0.1, center[0] + radius + 0.1)
    ax_main.set_ylim(center[1] - radius - 0.1, center[1] + radius + 0.1)

    # Create subplots for each emotion
    for idx, emotion in enumerate(emotion_cols):
        row = 1 + (idx // n_cols)  # Start from second row
        col = idx % n_cols
        ax = plt.subplot2grid((n_rows, n_cols), (row, col))
        
        # Plot points for this emotion
        mask = df['dominant_emotion'] == emotion
        count = mask.sum()  # Count the number of samples for this emotion
        
        # Separate by gender and plot with different colors
        female_mask = mask & (df['Gender'] == 'Female')
        male_mask = mask & (df['Gender'] == 'Male')
        unknown_mask = mask & (df['Gender'] == 'Unknown')
        
        # Plot females in pink
        ax.scatter(df.loc[female_mask, 'EmoVal'], 
                df.loc[female_mask, 'EmoAct'],
                color='#ffc8dd',
                label='Female',
                alpha=1.0)
        
        # Plot males in blue
        ax.scatter(df.loc[male_mask, 'EmoVal'], 
                df.loc[male_mask, 'EmoAct'],
                color='#bde0fe',
                label='Male',
                alpha=0.7)
        
        # Plot unknown in gray
        ax.scatter(df.loc[unknown_mask, 'EmoVal'], 
                df.loc[unknown_mask, 'EmoAct'],
                color='#888888',
                label='Unknown',
                alpha=0.7)
        
        # Add circular grid
        for r in np.linspace(0, radius, 6):
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            ax.plot(x, y, 'gray', alpha=0.3)

        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x = [center[0], center[0] + radius * np.cos(angle)]
            y = [center[1], center[1] + radius * np.sin(angle)]
            ax.plot(x, y, 'gray', alpha=0.3)
        
        ax.set_title(f'{emotion.capitalize()} Distribution (n={count})')
        ax.set_xlabel('EmoVal')
        ax.set_ylabel('EmoAct')
        ax.legend()
        ax.axis('equal')
        ax.set_xlim(center[0] - radius - 0.1, center[0] + radius + 0.1)
        ax.set_ylim(center[1] - radius - 0.1, center[1] + radius + 0.1)

    plt.tight_layout()
    plt.show()


def apply_dbscan_clustering(df, emotion, eps=0.01, min_samples=15):
    """
    Apply DBSCAN clustering to a specific emotion and return the clustered data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing emotion data
    emotion : str
        The emotion to analyze (e.g., 'neutral')
    eps : float
        The maximum distance between two samples to be considered as part of the same cluster
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point
        
    Returns:
    --------
    emotion_data : pandas.DataFrame
        The original data with cluster labels added
    X : pandas.DataFrame
        The features used for clustering
    clusters : numpy.ndarray
        The cluster labels for each sample
    """
    # Filter data for the specified emotion
    emotion_data = df[df['dominant_emotion'] == emotion].copy()
    
    # Select only EmoVal and EmoAct features for clustering
    features = ['EmoVal', 'EmoAct']
    X = emotion_data[features]
    
    # Check data size and adjust parameters if needed
    sample_count = len(X)
    print(f"Number of samples for '{emotion}': {sample_count}")
    
    # For large datasets (like 60,000 samples), DBSCAN can be memory-intensive
    # Consider using a sample if too large
    # if sample_count > 20000:
    #     print(f"Warning: Large dataset detected ({sample_count} samples). Using a 10% random sample for clustering.")
    #     X = X.sample(frac=0.1, random_state=42)
    #     print(f"Reduced to {len(X)} samples")
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)  # Use fit_predict instead of just fit
    
    # Calculate noise points ratio
    noise_points = np.sum(clusters == -1)
    total_points = len(clusters)
    noise_ratio = noise_points / total_points
    
    # Count number of clusters (excluding noise points which are labeled as -1)
    num_clusters = len(np.unique(clusters[clusters != -1]))
    
    
    # Add cluster labels to the data
    emotion_data['cluster'] = clusters
    
    # Print clustering statistics
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"\nClustering Statistics for {emotion}:")
    print(f"Noise points ratio: {noise_ratio:.2%} ({noise_points}/{total_points})")
    print(f"Number of clusters: {num_clusters}")
    print(f"Total points: {len(clusters)}")
    
    return emotion_data, X ,clusters

def plot_emotion_clusters(df, emotion, eps=0.1, min_samples=15):
    """
    Apply DBSCAN clustering to a specific emotion and visualize the results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing emotion data
    emotion : str
        The emotion to analyze (e.g., 'neutral')
    eps : float
        The maximum distance between two samples to be considered as part of the same cluster
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point
    """
    # Apply clustering
    emotion_data, X, clusters = apply_dbscan_clustering(df, emotion, eps, min_samples)
    
    # Create a figure with two subplots of equal width
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1]})
    
    # Plot 1: Original data (first two features)
    scatter1 = ax1.scatter(X.iloc[:, 0], X.iloc[:, 1], c='blue', alpha=0.5)
    ax1.set_title(f'Original {emotion} Data')
    ax1.set_xlabel('EmoVal')
    ax1.set_ylabel('EmoAct')
    
    # Set consistent axis limits for both plots
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Add circular grid to first plot
    center = (0.5, 0.5)
    radius = 0.5
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Draw concentric circles
    for r in np.linspace(0, radius, 6):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        ax1.plot(x, y, 'gray', alpha=0.3)
    
    # Draw radial lines
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x = [center[0], center[0] + radius * np.cos(angle)]
        y = [center[1], center[1] + radius * np.sin(angle)]
        ax1.plot(x, y, 'gray', alpha=0.3)
    
    # Plot 2: Clustered data
    scatter2 = ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    ax2.set_title(f'DBSCAN Clustering of {emotion} Data')
    ax2.set_xlabel('EmoVal')
    ax2.set_ylabel('EmoAct')
    
    # Set consistent axis limits for both plots
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Add circular grid to second plot
    # Draw concentric circles
    for r in np.linspace(0, radius, 6):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        ax2.plot(x, y, 'gray', alpha=0.3)
    
    # Draw radial lines
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x = [center[0], center[0] + radius * np.cos(angle)]
        y = [center[1], center[1] + radius * np.sin(angle)]
        ax2.plot(x, y, 'gray', alpha=0.3)
    
    # Add legend for clusters with discrete colors
    # Get unique cluster labels
    unique_clusters = np.unique(clusters)
    # Create a discrete colormap with distinct colors for better separation
    cmap = plt.colormaps['tab10'].resampled(len(unique_clusters))
    # Update the scatter plot with discrete colors
    scatter2.set_cmap(cmap)
    
    # Create legend handles and labels
    handles = []
    labels = []
    for i, cluster_id in enumerate(unique_clusters):
        color = cmap(i)
        patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                          markersize=10, label=f'Cluster {int(cluster_id)}')
        handles.append(patch)
        labels.append(f'Cluster {int(cluster_id)}')
    
    # Add legend to the plot
    ax2.legend(handles=handles, labels=labels, title='Clusters', 
              loc='best', bbox_to_anchor=(1.05, 1), frameon=True)
    
    # Ensure both plots have the same aspect ratio
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return emotion_data