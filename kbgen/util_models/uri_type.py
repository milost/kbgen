from ..util_models import URIEntity


class URIType(URIEntity):
    prefix = "http://dws.uni-mannheim.de/synthesized/Type_"

    def __init__(self, entity_id):
        super(URIType, self).__init__(entity_id)
