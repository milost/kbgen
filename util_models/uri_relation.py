from util_models import URIEntity


class URIRelation(URIEntity):
    prefix = "http://dws.uni-mannheim.de/synthesized/relation_"

    def __init__(self, entity_id):
        super(URIRelation, self).__init__(entity_id)
