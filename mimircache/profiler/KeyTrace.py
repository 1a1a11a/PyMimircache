import matplotlib.pyplot as plt


class KeyTrace:
    def __init__(self, reader, num_keys):
        self.reader = reader
        self.num_keys = num_keys

    def show(self, num_colors=4):
        x, y = self.generate_plot_data()

        colors = []
        color = 0
        colors.append(color)

        for e in range(1, len(y)):
            if y[e - 1] != y[e]:
                color += 1
                if color == num_colors:
                    color = 0

            colors.append(color)

        plt.scatter(x, y, marker='s', c=colors, edgecolors="white", alpha=0.5)
        plt.xlabel("Time")
        plt.ylabel("Key")
        plt.show()

    def generate_plot_data(self):
        logical_time = 0
        d = {}

        for element in self.reader:

            if element in d:
                d[element].append(logical_time)
            elif len(d) < self.num_keys:
                d[element] = []

            logical_time += 1

        x = []
        y = []
        key = 0

        for item in d.values():
            for time in item:
                x.append(time)
                y.append(key)

            key += 1

        return x, y
