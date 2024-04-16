class ByteTrie:
    """A python implementation mirroring the C++ ByteTrie class."""

    def __init__(self, byte_strings=None, values=None, parent=None):
        self._parent = parent
        self.match_version = -1
        self.match = False
        self.partial_match = False
        self.prob = 0
        self.value = -1
        self.children = {}

        if byte_strings is not None:
            if values is None:
                for s in byte_strings:
                    self.insert(s, 0)
            else:
                for i, s in enumerate(byte_strings):
                    self.insert(s, values[i])

    def keys(self):
        return self.children.keys()

    def has_child(self, byte):
        return byte in self.children

    def child(self, byte):
        return self.children[byte]

    def parent(self):
        return self._parent

    def size(self):
        return len(self.children)

    def __len__(self):
        return self.size()

    def insert(self, s, value, pos=0):
        if len(s) <= pos:
            if self.value < 0:
                self.value = value
        else:
            first_byte = s[pos : pos + 1]
            if first_byte not in self.children:
                self.children[first_byte] = ByteTrie(parent=self)
            self.children[first_byte].insert(s, value, pos + 1)

    def compute_probs(self, probs):
        self.prob = 0.0

        if self.value != -1:
            self.prob += probs[self.value]

        if self.children:
            for k in self.children:
                child = self.children[k]
                child.compute_probs(probs)
                self.prob += child.prob
