#converts brat annotations into JSON to develop relation extraction system for biomedical domain.
#does sentence splitting and tokenization using scispacy (which is suitable for bio domain).

#full-working version that puts cross-sentence AND inner-sentence things (entity, relations, triggers, events) on top of documents,
# Nothing inside corresponding sentences.

import sys
import os
import json
import spacy

class brat_json_Converter:
    def __init__(self, lp, program_halt, configs, scispacy_model="en_core_sci_sm"):
        self.lp = lp
        self.program_halt = program_halt
        self.configs = configs
        try:
            self.lp(["initializing brat_json_Converter.", "loading spacy and model: " + scispacy_model])
            self.nlp = spacy.load(scispacy_model, disable=["tagger", "ner"])
            self.lp("loading successful.")

        except Exception as E:
            msg = "unable to load spacy and/or the specified model. Error:\n" + str(E)
            self.program_halt(msg)

    def check_file_exists(self, file_address):
        if not os.path.isfile(file_address):
            self.program_halt("invalid file address: " + file_address)

    def __split_sentences (self, document_content):
        #receives document_content as have been previously read directly from a file and then runs scispacy sentence splitter and returns sentences with their offsets as a dictionary
        #keys of the dictionary are s0, s1, s2 , ...
        document = self.nlp(document_content)
        sentences = {}
        for sentence_id, sentence in enumerate(document.sents):
            sentences["s" + str(sentence_id)] = {
                "text" : sentence.text ,
                "bgn"  : sentence.start_char ,
                "end"  : sentence.end_char ,
            }
            """
                "entities" : {},
                "equivs"   : {},
                "triggers" : {},
                "events"   : {},
                "relations": {}
            }
            """
        return sentences

    def __get_brat_txt_info(self, input_file_address_txt, encoding ="utf-8"):
        #read brat .txt file, run sentence splitting on the text, returns sentences.
        with open(input_file_address_txt , "rt" , encoding=encoding) as f:
            document_content = f.read()
        sentences = self.__split_sentences(document_content)
        sentence_indices = {}
        for sentence_id in sorted(sentences.keys() , key = lambda x:int(x[1:])):
            sentence_indices[(sentences[sentence_id]['bgn'] , sentences[sentence_id]['end'])] = sentence_id
        return document_content, sentences , sentence_indices

    def __get_brat_ann_info(self, input_file_address_ann, all_event_types):
        #read brat .ann file, get information from file. returns dictionaries.
        ann_entities = {}
        ann_triggers = {}
        ann_events = {}
        ann_relations = {}
        ann_equivs = []
        ann_entity_attribs   = {}
        ann_event_attribs    = {}
        ann_relation_attribs = {}

        with open(input_file_address_ann, "r" , encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                res = line.split("\t")

                if res[0][0] == "#": #AnnotatorNotes  ... example: "#1	AnnotatorNotes T2	This is a note ..."
                    continue

                elif res[0][0] == "T": #ENTITIES and TRIGGERS
                    entity_id , entity_type_spans , entity_text = res
                    entity_type = entity_type_spans.split(" ")[0]
                    entity_spans= " ".join(entity_type_spans.split(" ")[1:])
                    entity_spans= [(int(x.split(" ")[0]), int(x.split(" ")[1])) for x in entity_spans.split(";")]
                    if not entity_type in all_event_types:
                        ann_entities[entity_id] = {"tag" : entity_type , "orig_spans" : entity_spans , "text": entity_text , "attribs" :{} ,"sentence_id" : None}
                    else:
                        ann_triggers[entity_id] = {"tag" : entity_type , "orig_spans" : entity_spans , "text": entity_text , "sentence_id" : None}

                elif res[0][0] == "E": #EVENTS
                    event_id = res[0]
                    event_info = res[1]
                    event_arguments = [(x.split(":")[0],x.split(":")[1]) for x in event_info.split(" ")]
                    ann_events[event_id] = {"arguments" : event_arguments , "attribs" : {} , "sentence_id" : None}

                elif res[0][0] == "R": #Relations ... example: "R1	Complex_formation Arg1:T4 Arg2:T5"
                    relation_id = res[0] # --> 'R1'
                    relation_info = res[1] # --> 'Complex_formation Arg1:T4 Arg2:T5'
                    relation_type = relation_info.split(" ")[0] # --> 'Complex_formation'
                    relation_arguments = [(x.split(":")[0],x.split(":")[1]) for x in relation_info.split(" ")[1:]]
                    ann_relations[relation_id] = {"type" : relation_type , "arguments" : relation_arguments , "attribs" : {} , "sentence_id" : None}

                elif res[0][0] == "*": #Equiv ... example: "*	Equiv T3 T4 T5"
                    if not "equiv" in res[1].lower():
                        continue
                    equivs = res[1].split(" ")[1:]
                    ann_equivs.append(equivs)

                elif res[0][0] == "A": #ARRTIBUTES
                    # Attribute for event  : "A45	Binding E5 BindingAttrib1"
                    # Arrtibute for entity : "A37	EntityAttrib T51 EntityAttrib1"
                    # Arrtibute for entity : "A32	Fusion T51"

                    attrib_id , attrib_info = res

                    attrib_info_split = attrib_info.split(" ")
                    if len(attrib_info_split) == 3:
                        attrib_name , related_entity_or_event_or_relation_id , attrib_value = attrib_info_split
                    elif len(attrib_info_split) == 2:
                        attrib_name, related_entity_or_event_or_relation_id = attrib_info_split
                        attrib_value = None

                    if related_entity_or_event_or_relation_id[0] == "T":
                        ann_entity_attribs[attrib_id] = {"entity_id": related_entity_or_event_or_relation_id, "name": attrib_name, "value": attrib_value}

                    elif related_entity_or_event_or_relation_id[0] == "E":
                        ann_event_attribs[attrib_id] = {"event_id": related_entity_or_event_or_relation_id, "name": attrib_name, "value": attrib_value}

                    elif related_entity_or_event_or_relation_id[0] == "R":
                        ann_relation_attribs[attrib_id] = {"relation_id": related_entity_or_event_or_relation_id, "name": attrib_name, "value": attrib_value}

                    else:
                        self.program_halt("error in Attribute: " + line)

                else:
                    self.program_halt("invalid line in brat ann :" + str(res))

        #sanity check-1: check all entities for the events exist
        for event_id in sorted(ann_events.keys() , key = lambda x: int(x[1:])):
            for index, entitytype_entityid in enumerate(ann_events[event_id]["arguments"]):
                entity_type , entity_id = entitytype_entityid
                if index == 0:
                    if not entity_id in ann_triggers:
                        self.program_halt("corresponding trigger " + entity_id + " for event " + event_id + " not found.")
                else:
                    if not entity_id in ann_entities:
                        self.program_halt("corresponding entity " + entity_id + " for event " + event_id + " not found.")

        #sanity check-2: check all entities for the relations exist
        for relation_id in sorted(ann_relations.keys() , key = lambda x: int(x[1:])):
            for entitytype_entityid in ann_relations[relation_id]["arguments"]:
                entity_type , entity_id = entitytype_entityid
                if not entity_id in ann_entities:
                    self.program_halt("corresponding entity " + entity_id + " for relation " + relation_id + " not found.")

        #sanity check-3: check all entities for Equivs exist
        for one_equiv_set in ann_equivs:
            for entity_id in one_equiv_set:
                if not entity_id in ann_entities:
                    self.program_halt("corresponding entity " + entity_id + " for equiv set " + str(one_equiv_set) + " not found.")

        #sanity check-4: check all entities for attribs exist + assign attribs to entities
        for attrib_id in sorted(ann_entity_attribs.keys() , key = lambda x: int(x[1:])):
            entity_id = ann_entity_attribs[attrib_id]['entity_id']
            if not entity_id in ann_entities:
                self.program_halt("corresponding entity " + entity_id + " for attribute " + attrib_id + " not found.")
            else:
                ann_entities[entity_id]["attribs"][attrib_id] = ann_entity_attribs[attrib_id]

        #sanity check-5: check all events for attribs exist + assign attribs to events
        for attrib_id in sorted(ann_event_attribs.keys() , key = lambda x: int(x[1:])):
            event_id = ann_event_attribs[attrib_id]['event_id']
            if not event_id in ann_events:
                self.program_halt("corresponding event " + event_id + " for attribute " + attrib_id + " not found.")
            else:
                ann_events[event_id]["attribs"][attrib_id] = ann_event_attribs[attrib_id]

        #sanity check-6: check all relations for attribs exist + assign attribs to relations
        for attrib_id in sorted(ann_relation_attribs.keys() , key = lambda x: int(x[1:])):
            relation_id = ann_relation_attribs[attrib_id]['relation_id']
            if not relation_id in ann_relations:
                self.program_halt("corresponding relation " + relation_id + " for attribute " + attrib_id + " not found.")
            else:
                ann_relations[relation_id]["attribs"][attrib_id] = ann_relation_attribs[attrib_id]

        #sort equivs:
        # example: [['T18' , 'T17' , 'T16']  , ['T5' , 'T6']] -->  [ ['T5' , 'T6'] , ['T16' , 'T17' , 'T18'] ]
        ann_equivs = [sorted(i) for i in ann_equivs]
        ann_equivs = sorted(ann_equivs, key=lambda x:int(x[0][1:]))

        return ann_entities, ann_triggers, ann_events, ann_relations, ann_equivs, ann_entity_attribs, ann_event_attribs ,ann_relation_attribs

    def convert_brat_to_python_dictionaries(self, input_file_address_txt, input_file_address_ann, all_event_types = [] , encoding ="utf-8"):
        self.check_file_exists(input_file_address_txt)
        self.check_file_exists(input_file_address_ann)

        ann_entities, ann_triggers, ann_events, ann_relations, ann_equivs, ann_entity_attribs, ann_event_attribs ,ann_relation_attribs = self.__get_brat_ann_info(input_file_address_ann, all_event_types)
        doc , sentences, sentence_indices = self.__get_brat_txt_info(input_file_address_txt, encoding=encoding)

        #calculate relative offsets for entities and triggers
        for _dict in [ann_entities , ann_triggers]:
            for entity_id in _dict.keys():
                constructed_entity_text_sentence = ""
                constructed_entity_text_document = ""
                sentence_ids = set()
                _dict[entity_id]['spans'] = []
                for original_entity_span in _dict[entity_id]['orig_spans']:
                    orig_span_bgn = original_entity_span[0] #bgn of this span
                    orig_span_end = original_entity_span[1] #end of this span
                    for sentence_span in sorted(sentence_indices.keys() , key= lambda x:x[0]):
                        sentence_bgn , sentence_end = sentence_span
                        if (orig_span_bgn >= sentence_bgn) and (orig_span_end <= sentence_end):
                            sentence_id = sentence_indices[sentence_span]
                            sentence_ids.add(sentence_id)
                            _dict[entity_id]['spans'].append((orig_span_bgn - sentence_bgn, orig_span_end - sentence_bgn))
                            if len(constructed_entity_text_sentence) == 0:
                                constructed_entity_text_sentence += sentences[sentence_id]['text'][orig_span_bgn - sentence_bgn:orig_span_end - sentence_bgn]
                            else:
                                constructed_entity_text_sentence += " " + sentences[sentence_id]['text'][orig_span_bgn - sentence_bgn:orig_span_end - sentence_bgn]

                            if len(constructed_entity_text_document) == 0:
                                constructed_entity_text_document += doc[orig_span_bgn:orig_span_end]
                            else:
                                constructed_entity_text_document += " " + doc[orig_span_bgn:orig_span_end]
                entity_text = _dict[entity_id]['text']
                _dict[entity_id]['sentence_id'] = sorted(sentence_ids, key = lambda x:int(x[1:]))

                problem1 = False
                problem2 = False
                if (constructed_entity_text_document != entity_text):
                    problem1 = True

                if (constructed_entity_text_sentence != entity_text):
                    problem2 = True

                if problem1 or problem2:
                    problem_msg = []
                    if problem1:
                        problem_msg += ["[WARNING] constructed_entity_text_document != entity_text", entity_id, constructed_entity_text_document, entity_text]
                    if problem2:
                        problem_msg += ["[WARNING] constructed_entity_text_sentence != entity_text", entity_id, constructed_entity_text_sentence, entity_text]
                    self.program_halt(problem_msg)

        #------------------------- FIND IF Cross or Inner sentence ? ---------------------------------------------------
        cross_sentence_entities  = {}
        cross_sentence_triggers  = {}
        cross_sentence_events    = {}
        cross_sentence_relations = {}

        # calculate and assign entities/triggers to sentences
        for _dict in [ann_entities , ann_triggers]:
            for entity_id in _dict.keys():
                if len(_dict[entity_id]['sentence_id']) > 1:
                    if _dict == ann_entities:
                        cross_sentence_entities[entity_id] = _dict[entity_id]
                    else:
                        cross_sentence_triggers[entity_id] = _dict[entity_id]
                """
                #uncomment if you want to assign inner-sentence entities/triggers also to their corresponding sentences. 
                else:
                    if _dict == ann_entities:
                        sentences[_dict[entity_id]['sentence_id'][0]]["entities"][entity_id] = _dict[entity_id]
                    else:
                        sentences[_dict[entity_id]['sentence_id'][0]]["triggers"][entity_id] = _dict[entity_id]
                """

        # calculate and assign events to sentences
        for event_id in ann_events.keys():
            this_event_all_related_sentences = set()
            for idx , entity_type_entity_id in enumerate(ann_events[event_id]['arguments']):
                entity_type , entity_id = entity_type_entity_id
                if idx==0: #first argument of event is always the trigger word ...
                    for s_id in ann_triggers[entity_id]['sentence_id']:
                        this_event_all_related_sentences.add(s_id)
                else:
                    for s_id in ann_entities[entity_id]['sentence_id']:
                        this_event_all_related_sentences.add(s_id)

            ann_events[event_id]['sentence_id'] = sorted(this_event_all_related_sentences, key=lambda x: int(x[1:]))

            if len(this_event_all_related_sentences) > 1:
                cross_sentence_events[event_id] = ann_events[event_id]
            """
            #uncomment if you want to assign inner-sentence events to their corresponding sentences 
            else:
                s_id = list(this_event_all_related_sentences)[0]
                sentences[s_id]["events"][event_id] = ann_events[event_id]
            """

        # calculate and assign relations to sentences
        for relation_id in ann_relations.keys():
            this_relation_all_related_sentences = set()
            for entity_type_entity_id in ann_relations[relation_id]['arguments']:
                entity_type , entity_id = entity_type_entity_id
                for s_id in ann_entities[entity_id]['sentence_id']:
                    this_relation_all_related_sentences.add(s_id)

            ann_relations[relation_id]['sentence_id'] = sorted(this_relation_all_related_sentences, key=lambda x: int(x[1:]))

            if len(this_relation_all_related_sentences) > 1:
                cross_sentence_relations[relation_id] = ann_relations[relation_id]
            """
            #uncomment if you want to assign inner-sentence relations to their corresponding sentences
            else:
                s_id = list(this_relation_all_related_sentences)[0]
                sentences[s_id]["relations"][relation_id] = ann_relations[relation_id]
            """

        return {
            "doc": doc,
            "sentences": sentences,
            "sentence_indices": sentence_indices,
            "ann_entities" : ann_entities,
            "ann_triggers" : ann_triggers,
            "ann_equivs"   : ann_equivs,
            "ann_events"   : ann_events,
            "ann_relations": ann_relations,
            "ann_entity_attribs"  : ann_entity_attribs,
            "ann_event_attribs"   : ann_event_attribs,
            "ann_relation_attribs": ann_relation_attribs,
            "cross_sentence_entities" : cross_sentence_entities,
            "cross_sentence_triggers" : cross_sentence_triggers,
            "cross_sentence_events"   : cross_sentence_events,
            "cross_sentence_relations": cross_sentence_relations,
        }

    def convert_brat_to_json(self, input_file_address_txt, input_file_address_ann, output_file_json, all_event_types=[], encoding ="utf-8"):
        json_content = {}
        json_content["metadata"] = {}
        json_content["documents"] = []

        result = self.convert_brat_to_python_dictionaries(input_file_address_txt, input_file_address_ann, all_event_types, encoding =encoding)

        document = {"id":"d0",
                    "metadata": {
                        "input_file_address_ann" : input_file_address_ann,
                        "input_file_address_txt" : input_file_address_txt,
                    },
                    "text": result['doc'],
                    "entities"  : result["ann_entities"],
                    "triggers"  : result["ann_triggers"],
                    "events"    : result["ann_events"],
                    "relations" : result["ann_relations"],
                    "equivs"    : result["ann_equivs"],
                    "sentences": []}

        for sentence_id in sorted(result['sentences'].keys() , key=lambda x: int(x[1:])):
            s = {"id": sentence_id ,
                 "text": result['sentences'][sentence_id]['text'],
                 "bgn" : result['sentences'][sentence_id]['bgn'],
                 "end" : result['sentences'][sentence_id]['end']
                 }
                 #"entities"  : result['sentences'][sentence_id]['entities'],
                 #"triggers"  : result['sentences'][sentence_id]['triggers'],
                 #"events"    : result['sentences'][sentence_id]['events']  ,
                 #"relations" : result['sentences'][sentence_id]['relations']
                 #}
            document['sentences'].append(s)

        json_content["documents"].append(document)

        if output_file_json is not None:
            try:
                with open(output_file_json, "w", encoding='utf-8') as jsonfile:
                        json.dump(json_content, jsonfile, ensure_ascii=False)
                    #self.lp("output: " + output_file_json)
            except Exception as E:
                self.program_halt("cannot create json file. Error: " + str(E))
        return json_content

    def has_duplicate_relation(self, brat_ann_filepath, all_event_types=[]):
        self.check_file_exists(brat_ann_filepath)
        ann_entities, ann_triggers, ann_events, ann_relations, ann_equivs, ann_entity_attribs, ann_event_attribs, ann_relation_attribs = self.__get_brat_ann_info(brat_ann_filepath, all_event_types)
        a1 = [frozenset([v['type']]+[i[1] for i in v['arguments']]) for k,v in ann_relations.items()]
        a2 = set(a1)
        return len(a1)!=len(a2)


