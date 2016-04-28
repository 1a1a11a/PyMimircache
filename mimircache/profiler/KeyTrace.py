import matplotlib.pyplot as plt


class KeyTrace:
    def __init__(self, reader, num_keys):
        self.reader = reader
        self.num_keys = num_keys

    def show(self, num_colors=4, min_frequency=0):
        x, y = self.generate_plot_data(min_frequency=min_frequency)

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

    def generate_plot_data(self, min_frequency=0):
        if min_frequency < 0:
            raise Exception("Invalid min_frequency: value must be positive")

        logical_time = 0
        d = {}

        values = None

        if min_frequency > 0:
            for element in self.reader:
                if element in d:
                    d[element].append(logical_time)
                else:
                    d[element] = []

                logical_time += 1

            for key, value in list(d.items()):
                if len(value) < min_frequency:
                    del d[key]

            values = sorted(d.values(), key=len)[len(d) - 1 - self.num_keys:len(d) - 1]

        else:
            for element in self.reader:

                if element in d:
                    d[element].append(logical_time)
                elif len(d) < self.num_keys:
                    d[element] = []

                logical_time += 1

            values = d.values()

        x = []
        y = []
        key = 0

        for item in values:
            for time in item:
                x.append(time)
                y.append(key)

            key += 1

        return x, y
