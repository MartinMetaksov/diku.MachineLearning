class DataSet(object):
    def __init__(self, entries):
        self.entries = entries

    def mutate_one_column(self, col, fun):
        for entry in self.entries:
            entry.vals[col] = fun(entry.vals[col])

    def mutate_all_columns(self, fun):
        self.entries = list(map(lambda x: DataEntry(list(map(lambda y: fun(y), x.vals)), x.ref), self.entries))

    def len(self):
        return self.entries.__len__()


class DataEntry:
    def __init__(self, vals, ref):
        self.vals = vals
        self.ref = ref
