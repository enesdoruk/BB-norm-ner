import copy


def lowercaser_mentions(dd_mentions):
    dd_lowercasedMentions = copy.deepcopy(dd_mentions)
    for id in dd_lowercasedMentions.keys():
        dd_lowercasedMentions[id]["mention"] = dd_mentions[id]["mention"].lower()
    return dd_lowercasedMentions


def lowercaser_ref(dd_ref):
    dd_lowercasedRef = copy.deepcopy(dd_ref)
    for cui in dd_ref.keys():
        dd_lowercasedRef[cui]["label"] = dd_ref[cui]["label"].lower()
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(tag.lower())
            dd_lowercasedRef[cui]["tags"] = l_lowercasedTags
    return dd_lowercasedRef