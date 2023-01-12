
class Templates(object):
    def __init__(self):
        self.template_list = []
        self.learning_rate = 0.1

    def computeMeanTemplate(self, spike, cluster):
        print(f"Cluster: {cluster}")
        print(f"Old Template List: {self.template_list}")
        if cluster >= len(self.template_list):
            self.template_list.append(spike)
            print(f"Added New Template in List: {self.template_list}")
        else:
            for dim in range(0, len(self.template_list[cluster])):
                old = self.template_list[cluster][dim]
                new = old + self.learning_rate*(spike[dim] - old)
                self.template_list[cluster][dim] = new
            print(f"New Template List: {self.template_list}")


# s = [1, 2, 3, 4, 5, 6]
# c = 0
# templates = Templates()
# templates.computeMeanTemplate(s, c)
# s = [4, 2, 3, 4, 5, 6]
# templates.computeMeanTemplate(s, c)

