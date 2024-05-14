import sys

def function_lp(msg):
    print(msg)

def function_program_halt(msg):
    print(msg)
    sys.exit(-1)

class Brat_ANN_Helper():
    def __init__(self, lp= None, program_halt= None):
        if lp is None:
            self.lp = function_lp
        else:
            self.lp = lp

        if program_halt is None:
            self.program_halt = function_program_halt
        else:
            self.program_halt = program_halt

    def read_ann(self, input_file_address, encoding="utf-8"):
        try:
            with open(input_file_address, "r", encoding=encoding) as f:
                file_content = [x.strip() for x in f.readlines() if len(x.strip())>0]
            return(file_content)
        except Exception as E:
            self.program_halt("cannot read file :" + input_file_address + "\nError: " + str(E))

    def get_brat_ann_info_from_file_conent(self, input_file_content, all_event_types=[]):
        # read brat .ann file content, get information from content, returns dictionaries.
        ann_entities = {}
        ann_triggers = {}
        ann_events = {}
        ann_relations = {}
        ann_equivs = []
        ann_entity_attribs = {}
        ann_event_attribs = {}
        ann_relation_attribs = {}
        ann_annotator_notes = {}

        for line in input_file_content:
            if len(line) == 0:
                continue
            line = line.strip()
            res = line.split("\t")

            if res[0][0] == "#":  # AnnotatorNotes  ... example: "#1	AnnotatorNotes T2	This is a note ..."
                annot_key = res[0]
                annot_ref = res[1].split(" ")[1]
                annot_txt = res[2]
                ann_annotator_notes[annot_key] = {'ref': annot_ref , 'txt': annot_txt}

            elif res[0][0] == "T":  # ENTITIES and TRIGGERS
                entity_id, entity_type_spans, entity_text = res
                entity_type = entity_type_spans.split(" ")[0]
                entity_spans = " ".join(entity_type_spans.split(" ")[1:])
                entity_spans = [(int(x.split(" ")[0]), int(x.split(" ")[1])) for x in entity_spans.split(";")]
                if not entity_type in all_event_types:
                    ann_entities[entity_id] = {"tag": entity_type, "orig_spans": entity_spans, "text": entity_text,
                                               "attribs": {}, "sentence_id": None}
                else:
                    ann_triggers[entity_id] = {"tag": entity_type, "orig_spans": entity_spans, "text": entity_text,
                                               "sentence_id": None}

            elif res[0][0] == "E":  # EVENTS
                event_id = res[0]
                event_info = res[1]
                event_arguments = [(x.split(":")[0], x.split(":")[1]) for x in event_info.split(" ")]
                ann_events[event_id] = {"arguments": event_arguments, "attribs": {}, "sentence_id": None}

            elif res[0][0] == "R":  # Relations ... example: "R1	Complex_formation Arg1:T4 Arg2:T5"
                relation_id = res[0]  # --> 'R1'
                relation_info = res[1]  # --> 'Complex_formation Arg1:T4 Arg2:T5'
                relation_type = relation_info.split(" ")[0]  # --> 'Complex_formation'
                relation_arguments = [(x.split(":")[0], x.split(":")[1]) for x in relation_info.split(" ")[1:]]
                ann_relations[relation_id] = {"type": relation_type, "arguments": relation_arguments, "attribs": {},
                                              "sentence_id": None}

            elif res[0][0] == "*":  # Equiv ... example: "*	Equiv T3 T4 T5"
                if not "equiv" in res[1].lower():
                    continue
                equivs = res[1].split(" ")[1:]
                ann_equivs.append(equivs)

            elif res[0][0] == "A":  # ARRTIBUTES
                # Attribute for event  : "A45	Binding E5 BindingAttrib1"
                # Arrtibute for entity : "A37	EntityAttrib T51 EntityAttrib1"
                # Arrtibute for entity : "A32	Fusion T51"

                attrib_id, attrib_info = res

                attrib_info_split = attrib_info.split(" ")
                if len(attrib_info_split) == 3:
                    attrib_name, related_entity_or_event_or_relation_id, attrib_value = attrib_info_split
                elif len(attrib_info_split) == 2:
                    attrib_name, related_entity_or_event_or_relation_id = attrib_info_split
                    attrib_value = None

                if related_entity_or_event_or_relation_id[0] == "T":
                    ann_entity_attribs[attrib_id] = {"entity_id": related_entity_or_event_or_relation_id,
                                                     "name": attrib_name, "value": attrib_value}

                elif related_entity_or_event_or_relation_id[0] == "E":
                    ann_event_attribs[attrib_id] = {"event_id": related_entity_or_event_or_relation_id,
                                                    "name": attrib_name, "value": attrib_value}

                elif related_entity_or_event_or_relation_id[0] == "R":
                    ann_relation_attribs[attrib_id] = {"relation_id": related_entity_or_event_or_relation_id,
                                                       "name": attrib_name, "value": attrib_value}

                else:
                    self.program_halt("error in Attribute: " + line)

            else:
                self.program_halt("invalid line in brat ann :", res)

        # sanity check-1: check all entities for the events exist
        for event_id in sorted(ann_events.keys(), key=lambda x: int(x[1:])):
            for index, entitytype_entityid in enumerate(ann_events[event_id]["arguments"]):
                entity_type, entity_id = entitytype_entityid
                if index == 0:
                    if not entity_id in ann_triggers:
                        self.program_halt("corresponding trigger " + entity_id + " for event " + event_id + " not found.")
                else:
                    if not entity_id in ann_entities:
                        self.program_halt("corresponding entity " + entity_id + " for event " + event_id + " not found.")

        # sanity check-2: check all entities for the relations exist
        for relation_id in sorted(ann_relations.keys(), key=lambda x: int(x[1:])):
            for entitytype_entityid in ann_relations[relation_id]["arguments"]:
                entity_type, entity_id = entitytype_entityid
                if not entity_id in ann_entities:
                    self.program_halt("corresponding entity " + entity_id + " for relation " + relation_id + " not found.")

        # sanity check-3: check all entities for Equivs exist
        for one_equiv_set in ann_equivs:
            for entity_id in one_equiv_set:
                if not entity_id in ann_entities:
                    self.program_halt("corresponding entity " + entity_id + " for equiv set " + str(one_equiv_set) + " not found.")

        # sanity check-4: check all entities for attribs exist + assign attribs to entities
        for attrib_id in sorted(ann_entity_attribs.keys(), key=lambda x: int(x[1:])):
            entity_id = ann_entity_attribs[attrib_id]['entity_id']
            if not entity_id in ann_entities:
                self.program_halt("corresponding entity " + entity_id + " for attribute " + attrib_id + " not found.")
            else:
                ann_entities[entity_id]["attribs"][attrib_id] = ann_entity_attribs[attrib_id]

        # sanity check-5: check all events for attribs exist + assign attribs to events
        for attrib_id in sorted(ann_event_attribs.keys(), key=lambda x: int(x[1:])):
            event_id = ann_event_attribs[attrib_id]['event_id']
            if not event_id in ann_events:
                self.program_halt("corresponding event " + event_id + " for attribute " + attrib_id + " not found.")
            else:
                ann_events[event_id]["attribs"][attrib_id] = ann_event_attribs[attrib_id]

        # sanity check-6: check all relations for attribs exist + assign attribs to relations
        for attrib_id in sorted(ann_relation_attribs.keys(), key=lambda x: int(x[1:])):
            relation_id = ann_relation_attribs[attrib_id]['relation_id']
            if not relation_id in ann_relations:
                self.program_halt("corresponding relation " + relation_id + " for attribute " + attrib_id + " not found.")
            else:
                ann_relations[relation_id]["attribs"][attrib_id] = ann_relation_attribs[attrib_id]

        # sort equivs:
        # example: [['T18' , 'T17' , 'T16']  , ['T5' , 'T6']] -->  [ ['T5' , 'T6'] , ['T16' , 'T17' , 'T18'] ]
        ann_equivs = [sorted(i) for i in ann_equivs]
        ann_equivs = sorted(ann_equivs, key=lambda x: int(x[0][1:]))

        return {"ann_entities" : ann_entities,
                "ann_triggers" : ann_triggers,
                "ann_events"   : ann_events,
                "ann_relations": ann_relations,
                "ann_equivs"   : ann_equivs,
                "ann_entity_attribs"   : ann_entity_attribs,
                "ann_event_attribs"    : ann_event_attribs,
                "ann_relation_attribs" : ann_relation_attribs,
                "ann_annotator_notes"  : ann_annotator_notes}

    def write_ann_file_string_corpus(self, ann_res, ann_output_file_address):
        try:
            ann_entities = ann_res['ann_entities'] #{}
            ann_triggers = ann_res['ann_triggers'] #{}
            ann_events = ann_res['ann_events'] #{}
            ann_relations = ann_res['ann_relations'] #{}
            ann_equivs = ann_res['ann_equivs'] #[]
            ann_entity_attribs = ann_res['ann_entity_attribs'] #{}
            ann_event_attribs = ann_res['ann_event_attribs'] #{}
            ann_relation_attribs = ann_res['ann_relation_attribs'] #{}
            ann_annotator_notes = ann_res['ann_annotator_notes'] #{}
        except Exception:
            self.program_halt("not all fields found in the input dictionary.")

        try:
            ann_output_file_handle = open(ann_output_file_address , "wt" , encoding='utf-8')
        except Exception as E:
            self.program_halt("could not create output file: " + ann_output_file_address + "\nError: " + str(E))

        #print("this version only supports string-corpus '.ann' files. Do not use it for anything else. Many features not implemented.", file=sys.stderr)

        for equiv_set in ann_equivs:
            output = "*" + "\tEquiv "
            output+= " ".join(equiv_set) + "\n"
            ann_output_file_handle.write(output)

        for entity_id in ann_entities.keys():
            output = entity_id + "\t"
            output+= ann_entities[entity_id]['tag'] + " "
            output+= ";".join([' '.join([str(p) for p in i]) for i in ann_entities[entity_id]['orig_spans']])
            output+= "\t" + ann_entities[entity_id]['text'] + "\n"
            ann_output_file_handle.write(output)

        for attrib_id in ann_entity_attribs.keys():
            output = attrib_id + "\t"
            output+= ann_entity_attribs[attrib_id]['name'] + " " + ann_entity_attribs[attrib_id]['entity_id']   + "\n"
            ann_output_file_handle.write(output)

        for annot_id in ann_annotator_notes.keys():
            output = annot_id + "\tAnnotatorNotes "
            output+= ann_annotator_notes[annot_id]['ref'] + "\t" + ann_annotator_notes[annot_id]['txt'] + "\n"
            ann_output_file_handle.write(output)

        for rel_id in ann_relations:
            output = rel_id + "\t"
            output+= ann_relations[rel_id]['type'] + " "
            output+= " ".join([e1 + ":" + e2 for e1, e2 in ann_relations[rel_id]['arguments']])
            output+= "\n"
            ann_output_file_handle.write(output)










