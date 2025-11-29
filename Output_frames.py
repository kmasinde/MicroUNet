for i in range(3):  # show first 3 images in the batch
    plt.figure(figsize=(10,3))

    # Input image
    plt.subplot(1,3,1)
    plt.imshow(np.transpose(imgs[i].cpu().numpy(), (1,2,0)))
    plt.title("Input")
    plt.axis('off')

    # Ground truth mask
    plt.subplot(1,3,2)
    plt.imshow(masks[i,0].cpu().numpy(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    # Predicted mask
    plt.subplot(1,3,3)
    plt.imshow(preds_bin[i,0].cpu().numpy(), cmap='gray')
    plt.title("Prediction")
    plt.axis('off')

    plt.show()