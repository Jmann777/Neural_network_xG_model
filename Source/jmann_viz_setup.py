import matplotlib.pyplot as plt

# function to scale the image to desires length and with
def scale(im, nR, nC):
    """
    parameters
    im :image
    nR:row size
    nC:column size
    Return: scaled image
    """
    number_rows = len(im)  # source number of rows
    number_columns = len(im[0])  # source number of columns
    return [[im[int(number_rows * r / nR)][int(number_columns * c / nC)]
             for c in range(nC)] for r in range(nR)]


def watermark(ax, x0, y0):
    """
    adds image logo and positions it on the plot
    ax: figure object
    x0: adds x
    y0: adds y
    """
    logo = plt.imread('jmann-logo.png')
    # scale Image
    logo = scale(logo, 150, 150)

    ax.figure.figimage(logo, x0, y0, zorder=2, origin="upper")
