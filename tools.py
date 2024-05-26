from pronto import Ontology
import copy



def loader_ontobiotope(filePath):
    """
    Description: A loader of OBO ontology based on Pronto lib.
    (maybe useless...)
    :param filePath: Path to the OBO file.
    :return: an annotation ontology in a dict (format: concept ID (string): {'label': preferred tag,
    'tags': list of other tags, 'parents': list of parents concepts}
    """
    dd_obt = dict()
    onto = Ontology(filePath)
    for o_concept in onto:
        dd_obt[o_concept.id] = dict()
        dd_obt[o_concept.id]["label"] = o_concept.name
        dd_obt[o_concept.id]["tags"] = list()

        for o_tag in o_concept.synonyms:
            dd_obt[o_concept.id]["tags"].append(o_tag.desc)

        dd_obt[o_concept.id]["parents"] = list()
        for o_parent in o_concept.parents:
            dd_obt[o_concept.id]["parents"].append(o_parent.id)

    return dd_obt


def is_desc(dd_ref, cui, cuiParent):
    """
    Description: A function to get if a concept is a descendant of another concept.
    Here, only used to select a clean subpart of an existing ontology (see select_subpart_hierarchy method).
    """
    result = False
    if "parents" in dd_ref[cui].keys():
        if len(dd_ref[cui]["parents"]) > 0:
            if cuiParent in dd_ref[cui]["parents"]:  # Not working if infinite is_a loop (normally never the case!)
                result = True
            else:
                for parentCui in dd_ref[cui]["parents"]:
                    result = is_desc(dd_ref, parentCui, cuiParent)
                    if result:
                        break
    return result


def select_subpart_hierarchy(dd_ref, newRootCuis):
    """
    Description: By picking a single concept in an ontology, create a new sub ontology with this concept as root.
    Here, only used to select the habitat subpart of the Ontobiotope ontology.
    """
    dd_subpart = dict()
    for newRootCui in newRootCuis:
        dd_subpart[newRootCui] = copy.deepcopy(dd_ref[newRootCui])
        dd_subpart[newRootCui]["parents"] = []

        for cui in dd_ref.keys():
            if is_desc(dd_ref, cui, newRootCui):
                dd_subpart[cui] = copy.deepcopy(dd_ref[cui])

        # Clear concept-parents which are not in the descendants of the new root:
        for cui in dd_subpart.keys():
            dd_subpart[cui]["parents"] = list()
            for parentCui in dd_ref[cui]["parents"]:
                if is_desc(dd_ref, parentCui, newRootCui) or parentCui == newRootCui:
                    dd_subpart[cui]["parents"].append(parentCui)

    return dd_subpart