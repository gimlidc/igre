import matplotlib.pyplot as plt

verbose_level = 1  # display flag, tied to Verbose "enum"
#     always = -1       -  displays only things with always tag
#     metacentrum = 0   -  metacentrum setting, display almost nothing possibly dump into file
#     normal = 1        -
#     debug = 4         - display everything but things with never tag
#     never = 666       -

config = {}
shift_multi = 350
shift_multi_2 = 3500


class Verbose:
    # enum
    always = -1  # always, prints this
    metacentrum = 0
    normal = 1
    debug = 4
    never = 666

    # TODO: if level = metacentrum:
    #              dump_to_file()
    @staticmethod
    def print(a, level=1):
        if level <= verbose_level:
            print(a)

    @staticmethod
    def imshow(a, level=1):
        if level <= verbose_level:
            if (len(a.shape) > 2) and a.shape[2] == 1:
                plt.imshow(a.reshape(a.shape[0], a.shape[1]), cmap='gray')
            elif len(a.shape) == 2:
                plt.imshow(a, cmap='gray')
            else:
                plt.imshow(a[:,:,:3])
            plt.show()

    @staticmethod
    def plot(a, title=None, level=1):
        if level <= verbose_level:
            plt.plot(a)
            if title is not None:
                plt.title(title)
            plt.show()
