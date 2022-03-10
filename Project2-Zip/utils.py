import matplotlib.pyplot as plt

def display_images(x, N, color=False):
    fig = plt.figure(figsize=(20, 4))
    columns = 10
    rows = N//columns

    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        if color:
            plt.imshow(x[i-1])
        else:
            plt.imshow(x[i-1].reshape(28, 28))
        plt.axis('off')
    plt.show()
