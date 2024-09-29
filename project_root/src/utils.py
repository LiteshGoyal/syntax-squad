import matplotlib.pyplot as plt

def plot_images(images, labels, predictions=None, n=5):
    fig, axes = plt.subplots(1, n, figsize=(20, 4))
    for i in range(n):
        axes[i].imshow(images[i])
        title = f"True: {labels[i]}"
        if predictions is not None:
            title += f"\nPred: {predictions[i]}"
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()