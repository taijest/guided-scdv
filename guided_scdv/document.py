from typing import List

Document = List[List[str]]


class Documents:
    def __init__(self, documents: Document) -> None:
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        for document in self.documents:
            yield document
