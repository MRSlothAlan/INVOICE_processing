"""
Goal:
Use this to parse the invoice and generate data sets

Now only support english invoices, just to simplify things.
if not english: OCR slow

Only for Development
"""
import pytesseract
from tqdm import tqdm

from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.opencv_image_operations import resize_with_ratio, \
    draw_rectangle_text_with_ratio, pre_process_images_before_scanning, auto_align_image
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.information_finder import find_information_rule_based, \
    find_line_item_rule_based
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.feature_extraction.graph_construction import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.ALGO.region_proposal import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.OCR_operation import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.word_feature_calculation import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.feature_extraction.node_label_matrix import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.feature_extraction.basic_gcn_operation import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.word_parser.keyword_extraction import *
import os
import pickle
import time


def parse_main():
    """
    :return:
    """

    # LOAD MODEL
    """
    print("\n===== LOAD MODEL ======\n")
    get_pretrained_model()
    print("COMPLETE\n")
    wv.vocab
    wv.most_similar(positive=["invoice"], topn=5)
    """
    if not DL:
        image_files = get_image_files()
    elif DL:
        if GCN:
            print("=== The script now generate data set for GCN training ===")
            image_data_set_dir = Path.cwd().parents[1] / "0_RAW_DATASETS/all_raw_images"
            label_data_set_dir = Path.cwd().parents[1] / "0_RAW_DATASETS/all_raw_images_labels"
            processed_dir = Path.cwd() / "AI/GNN/processed_GCN"

            if not os.path.exists(str(processed_dir)):
                os.makedirs(str(processed_dir))
            class_list = load_class_list(label_data_set_dir=label_data_set_dir)
            # remember to save this class_dict for further usage
            image_files_temp = listdir(str(label_data_set_dir))
            image_files = [i for i in image_files_temp if
                            str(i).rsplit('.', 1)[1] != "txt"]
    else:
        image_files = get_image_files()

    # load csv of currency, save as dictionary
    currency_dict = get_currency_csv()
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    for index, image_name in enumerate(tqdm(image_files)):

        print("\n---------------------------------------------\n"
              "processing {}".format(image_name),
              "\n---------------------------------------------\n")

        start = time.time()
        image_path = str(dataset_dir / image_name)
        if DL:
            if GCN:
                label_file_name = image_name.rsplit('.', 1)[0] + ".xml"
                processed_data_set_dir = image_name.rsplit('.', 1)[0] + ".txt"
                image_name = image_name.rsplit('.', 1)[0] + ".jpg"

                image_path = str(image_data_set_dir / image_name)

        image = cv2.imread(image_path, 1)

        """
        17092020: try not to do any preprocessing
        """
        # image_pil = pre_process_images_before_scanning(image)
        # image = np.array(image_pil)
        # to OpenCV format
        # image = image[:, :, ::-1].copy()
        # align image

        if AUTO_ALIGN:
            print("auto align image...")
            image = auto_align_image(img=image)
            print("done")

        resize = resize_with_ratio(image, resize_ratio)
        resize_region = resize.copy()
        resize_copy = resize.copy()
        resize_temp = resize.copy()
        resize_function = resize.copy()
        resize_mst = resize.copy()

        end = time.time()
        if TIMER:
            print("load images and pre-process, time: {}".format(abs(start - end)))

        start = time.time()
        info = pytesseract.image_to_data(image, output_type='dict')
        image_copy = image.copy()
        end = time.time()
        if TIMER:
            print("OCR image, time: {}".format(abs(start - end)))

        start = time.time()
        """ index defined: 
         level, page_num, block_num, par_num
         line_num, word_num, left, top, width, height, conf, text
         """
        words_raw, same_line, same_block = ocr_to_standard_data(info)

        # used for node connections
        same_line_copy = same_line
        # same_line_copy = same_line

        # get total number of nodes
        total_number_of_nodes = len(words_raw) - 1
        """
        print("generating minimum spanning tree...")
        mst, starting_node_id = generate_mst_graph(glw_detailed, total_number_of_nodes)
        print("Done")

        if SHOW_IMAGE:
            resize_mst = mst.draw_mst_on_graph(words_raw, resize_mst, resize_ratio)
            cv2.imshow("MST", resize_mst)
        """
        if DEBUG:
            if SHOW_IMAGE or SHOW_SUB_IMAGE:
                resize_temp = same_line_copy.draw_graph(words_raw, resize_temp, resize_ratio)
        end = time.time()
        if TIMER:
            print("Process OCR data, time: {}".format(abs(start - end)))
        if DL:
            start = time.time()
            # 08092020: need to consider node distance in invoice. All data in processed must be rewritten
            print("generate graph for individual words (detailed)")
            glw_detailed = same_line_copy.generate_graph()
            print("Done")
            if GCN:
                # if processed_data_set_dir not in set(listdir(processed_dir)):
                """
                08092020
                === graph convolution network ===
                the functions below generates the 
                """
                # N = number of nodes
                print("=== GENERATE FEATURES FOR GCN ===")
                """
                binary label matrix
                size = N * E (size of labels)
                """
                len_set = set()
                print("     generate node label matrix")

                gcn_node_label = node_label_matrix(word_node=same_line.return_raw_node(),
                                                   classes=class_list,
                                                   label_data_set_dir=label_data_set_dir,
                                                   label_file_name=label_file_name)
                print("     SHAPE=", len(gcn_node_label))
                for key in gcn_node_label:
                    len_set.add(len(gcn_node_label[key]))
                print("     {}".format(len_set))
                """
                adjacency matrix
                size = N * N
                """
                print("     generate adjacency matrix...")
                height, width, color = image.shape
                gcn_adj_matrix = adjacency_matrix_basic(word_nodes=same_line.return_raw_node(), height=height, width=width)
                print("     SHAPE=", np.array(gcn_adj_matrix).shape)

                """
                return a list of node - feature list
                size = N * 109
                [[node_feature], [node-feature], [], ...]
                """
                print("     generate node feature list...")
                # a dictionary
                wf = WordFeature()
                raw = same_line_copy.return_raw_node()
                gcn_node_feature = wf.feature_calculation(same_line_copy.return_raw_node(), image=image)
                if DEBUG:
                    if DEBUG_DETAILED:
                        for index, feature in enumerate(gcn_node_feature):
                            print(raw[index].word)
                            print(gcn_node_feature[feature])
                print("     SHAPE=", len(gcn_node_feature))
                len_set = set()
                for key in gcn_node_feature:
                    len_set.add(len(gcn_node_feature[key]))
                print("     {}".format(len_set))

                # save all features to a file
                CLASS_LIST_LEN = len(class_list)
                NODE_SIZE = len(gcn_node_label)
                FEATURE_LEN = len(gcn_node_feature[0])

                data_to_save = [gcn_node_label, gcn_adj_matrix, gcn_node_feature, CLASS_LIST_LEN, NODE_SIZE, FEATURE_LEN]
                with open(str(processed_dir / processed_data_set_dir), "wb") as processed_f:
                    pickle.dump(data_to_save, processed_f)
                processed_f.close()
                end = time.time()
                if TIMER:
                    print("produced GCN data, time: {}".format(abs(start - end)))

        # need to merge some nodes which are closed together
        # word model will be applied here
        height, width, color = image.shape
        # need to update the merging method
        words_raw_new, all_raw_colon_separated_entry = same_line.merge_nearby_token(width)

        """
        16092020:
        try to extract keywords!
        """
        # keyword_extraction(words_raw_new)

        # for node in words_raw_new:
        # print(node.word)
        # need to plot a graph to check
        if DEBUG_DETAILED:
            if SHOW_IMAGE:
                for index, line_data in enumerate(words_raw_new):
                    resize = draw_rectangle_text_with_ratio(resize,
                                                            line_data.left, line_data.top,
                                                            line_data.width,
                                                            line_data.height,
                                                            ' ',
                                                            resize_ratio)
                cv2.imshow("merged nodes", resize)
                cv2.waitKey(0)
        print("generate graph of merged nodes")
        same_line.generate_graph()
        print("Done")

        # each node now has the connection data
        # draw the results
        # 20082020: logic error, need to get a list of raw, MERGED tokens
        resize_copy = same_line.draw_graph(words_raw_new, resize_copy, resize_ratio)

        json_name = output_json_dir / str(image_name[:-4] + ".json")

        # write parsed words to a text file
        """
        same_block.write_to_file(file_name)
        same_line.write_to_file(file_name)
        """

        if DEBUG:
            if SHOW_IMAGE or SHOW_SUB_IMAGE:
                # cv2.imshow("image", resize)
                cv2.imshow("graph", resize_copy)
                cv2.imshow("original graph", resize_temp)

        if PARSE:
            print("propose regions")
            # try to pre-process image, generate regions
            """
            format of entry:
            [x, y, w, h]
            """
            rect_regions = region_proposal(image)

            if SHOW_IMAGE:
                for rect in rect_regions:
                    cv2.rectangle(resize_region, (rect[0], rect[1]),
                                  (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)
                cv2.imshow("regions", resize_region)
            print("finish")

            import enchant
            d = enchant.Dict("en_US")

            # check specific entries only
            image, all_results = find_information_rule_based(words_raw_new, resize_function, resize_ratio, d)
            # may also need to pass raw nodes in order to split the content
            all_line_items = find_line_item_rule_based(words_raw_new, words_raw, rect_regions, resize_ratio, image_copy)
            # with the node structure, you can tag stuff easily.
            results, currency = same_line_copy.use_parser_re(currency_dict)
            if currency is not None:
                print(currency_dict[currency.upper()])
            else:
                print("currency undefined")
            # return a json file of data
            results = list()
            for tagged_items in all_results:
                label = tagged_items[0]
                node_entry = tagged_items[1]
                node_origin = tagged_items[2]
                # format [[left, top, width, height, 'invoice_number', original_word, entry]]
                # format ['HONG KONG', 'Hong Kong Dollar']
                # def save_as_json(json_path, results, currency, currency_info):
                results.append([node_entry.left,
                                node_entry.top,
                                node_entry.width,
                                node_entry.height,
                                label,
                                node_origin.word,
                                node_entry.word])
                # print(label, node_entry.word, node_origin.word)

            cv2.waitKey(0)
            try:
                save_as_json(json_name, results, all_line_items, currency, currency_dict[currency.upper()],
                             raw_colon_separated_entries=all_raw_colon_separated_entry)
            except AttributeError as e:
                save_as_json(json_name, results, all_line_items, currency=None, currency_info=None,
                             raw_colon_separated_entries=all_raw_colon_separated_entry)


if __name__ == '__main__':
    parse_main()

