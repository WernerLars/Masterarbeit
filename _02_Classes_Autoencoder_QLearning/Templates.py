
class Templates(object):
    def __init__(self):
        self.template_list = []
        self.learning_rate = 0.1

    def computeMeanTemplate(self, spike, cluster):
        """
            adds spike, if cluster label is bigger than length of template list
            otherwise computes mean template for every dimension of incoming spike features (only for specified cluster)
            mean is computed new-old with a learning rate to limit change of mean template
        """

        if cluster >= len(self.template_list):
            self.template_list.append(spike)
        else:
            for dim in range(0, len(self.template_list[cluster])):
                old = self.template_list[cluster][dim]
                new = old + self.learning_rate*(spike[dim] - old)
                self.template_list[cluster][dim] = new
