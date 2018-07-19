from util_models import URIEntity


class URIType(URIEntity):
    prefix = "http://dws.uni-mannheim.de/synthesized/Type_"

    def __init__(self, r):
        super(URIType, self).__init__(r)
