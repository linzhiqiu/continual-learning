# A script to parse flickr datasets/autotags
from PIL import Image
import requests
from io import BytesIO
import os
import json
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=[
                                "dataset", # yfcc100m dataset
                            ], help="The task to perform.")
parser.add_argument("--img_dir", 
                    default='/Users/linzhiqiu/Downloads/yfcc100m/data',
                    help="The yfcc100M dataset store location")
parser.add_argument("--data_file", 
                    default='/Users/linzhiqiu/Downloads/yfcc100m/yfcc100m_dataset',
                    help="The yfcc100M dataset file ")
parser.add_argument("--auto_file",
                    default='/Users/linzhiqiu/Downloads/yfcc100m/yfcc100m_autotags-v1', 
                    help="The autotag file")
parser.add_argument("--exif_file",
                    default='/Users/linzhiqiu/Downloads/yfcc100m/yfcc100m_exif', 
                    help="The exif file")
parser.add_argument("--hash_file",
                    default='/Users/linzhiqiu/Downloads/yfcc100m/yfcc100m_hash', 
                    help="The hash file")
parser.add_argument("--lines_file",
                    default='/Users/linzhiqiu/Downloads/yfcc100m/yfcc100m_lines', 
                    help="The lines file")
parser.add_argument("--size_option",
                    default='original', choices=['original'],
                    help="Whether to use the original image size (max edge has 500 px).")
parser.add_argument("--max_images",
                    type=int, default=50000,
                    help="The maximum images to store")
parser.add_argument("--original_size",
                    action='store_true',
                    help="Whether to use the original image size.")
parser.add_argument("--min_edge",
                    type=int, default=0,
                    help="Images with edge shorter than min_edge will be ignored.")
parser.add_argument("--min_size",
                    type=int, default=2100,
                    help="Images with size smaller than min_size will be ignored.")


# The index for dataset file
IDX_LIST = [
    "ID",
    "USER_ID",
    "NICKNAME",
    "DATE_TAKEN",
    "DATE_UPLOADED",
    "DEVICE",
    "TITLE",
    "DESCRIPTION",
    "USER_TAGS",
    "MACHINE_TAGS",
    "LON",
    "LAT",
    "GEO_ACCURACY",
    "PAGE_URL",
    "DOWNLOAD_URL",
    "LICENSE_NAME",
    "LICENSE_URL",
    "SERVER_ID",
    "FARM_ID",
    "SECRET",
    "SECRET_ORIGINAL",
    "EXT",
    "IMG_OR_VIDEO",
]

IDX_TO_NAME = {i : IDX_LIST[i] for i in range(len(IDX_LIST))}

NAME_TO_DESCRIPTION = {
     "ID" : "Photo/video identifier",
     "USER_ID" : "User NSID",
     "NICKNAME" : "User nickname",
     "DATE_TAKEN" : "Date taken",
     "DATE_UPLOADED" : "Date uploaded",
     "DEVICE" : "Capture device",
     "TITLE" : "Title",
     "DESCRIPTION" : "Description",
     "USER_TAGS" : "User tags (comma-separated)",
     "MACHINE_TAGS" : "Machine tags (comma-separated)",
     "LON" : "Longitude",
     "LAT" : "Latitude",
     "GEO_ACCURACY" : "Accuracy of the longitude and latitude coordinates (1=world level accuracy, ..., 16=street level accuracy)",
     "PAGE_URL" : "Photo/video page URL",
     "DOWNLOAD_URL" : "Photo/video download URL",
     "LICENSE_NAME" : "License name",
     "LICENSE_URL" : "License URL",
     "SERVER_ID" : "Photo/video server identifier",
     "FARM_ID" : "Photo/video farm identifier",
     "SECRET" : "Photo/video secret",
     "SECRET_ORIGINAL" : "Photo/video secret original",
     "EXT" : "Extension of the original photo",
     "IMG_OR_VIDEO" : "Photos/video marker (0 = photo, 1 = video)",
}



# The index for autotag file
autotag_ID = 0
autotag_CONCEPTS = 1

def fetch_and_save_image(img_path, url):
    number_of_trails = 0
    sleep_time = 5
    while True:
        try:
            # print(1)
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            if img.size[0] < args.min_edge or img.size[1] < args.min_edge:
                return False
            img.save(img_path)
            return True
        except:
            # Sleep for a while and try again
            number_of_trails += 1
            sleep_time += 10
            if number_of_trails >= 3:
                print("Cannot fetch {:s} after 3 trails".format(url))
                return False
            print("Sleep for {:d} secs".format(sleep_time))
            time.sleep(sleep_time)
            continue
        

class FlickrParser():
    def __init__(self, args):
        self.args = args
        self.data_file = args.data_file
        self.auto_file = args.auto_file
        self.exif_file = args.exif_file
        self.hash_file = args.hash_file
        self.lines_file = args.lines_file
        self.max_images = args.max_images
        self.size_option = args.size_option
        self.min_edge = args.min_edge
        self.min_size = args.min_size

        self._load_files()
        # # Parse by readDatasetFile
        # self.imageinfo_by_author_1 = {} # The author dict has (key=author, value=id)
        # self.imageinfo_by_author_2 = {}
        # self.imageinfo_by_id = {} # The id dict has (key=id, value=everything)

        # self.brokenImages = [] # A list of image id
        # # Parse by matchWithAutotagFile
        # self.imageautotag_by_id = {} # A nested dict (key=id, value=dict(key=concept, value=score))
        # self.visual_concepts = {} # A dict (key=concept, value=list of tuples (id, score))
        # self.max_images = max_images
        # self.image_with_no_tags = 0
        # self.all_concepts = {}

    def _parse_metadata(self, line):
        entries = line.split("\t")
        meta = {IDX_TO_NAME[i] : entries[i] for i in range(len(entries))}
        return meta
    
    def _parse_autotags(self, line):
        entries = line.split("\t")
        tags = entries[1].split(",")
        tag_scores = {t.split(":")[0] : float(t.split(":")[1]) for t in tags}
        return tag_scores

    def _load_files(self):
        metadata = []
        with open(self.data_file, "r") as f:
            with open(self.auto_file, "r") as auto_f:
                for f_line, auto_line in zip(f, auto_f):
                    meta = self._parse_metadata(f_line)
                    autotag_scores = self._parse_autotags(auto_line)
                    meta["AUTO_TAG_SCORES"] = autotag_scores
                    metadata.append(meta)
                    print(meta)
                    import pdb; pdb.set_trace()

    def remove_broken_path(self, img_path):
        """Remove an image locally if it's broken
        """
        try:
            a = Image.open(img_path)
        except:
            print(img_path + " is broken, so we delete it")
            os.remove(img_path)
            self.brokenImages += [img_path]
        if os.path.getsize(img_path) < 2100: #2051
            print(img_path + " is too small, so we delete it")
            os.remove(img_path)
            self.brokenImages += [img_path]

    def readDatasetFile(self, datasets_path):
        """Read from the dataset text file, add all unique author images.
        """
        with open(datasets_path, "r") as f:
            for line in f:
                if len(self.imageinfo_by_id) >= self.max_images:
                    break

                line = line.split()

                # print(len(line))
                
                # If author already in here, then skip
                if line[idx_AUTHOR_1] in self.imageinfo_by_author_1:
                    continue
                if line[idx_AUTHOR_2] in self.imageinfo_by_author_2:
                    continue
                if line[idx_URL].find("video") != -1:
                    # URL is a video
                    continue

                self.imageinfo_by_author_1[line[idx_AUTHOR_1]] = line[idx_ID]
                self.imageinfo_by_author_2[line[idx_AUTHOR_2]] = line[idx_ID]
                self.imageinfo_by_id[line[idx_ID]] = line

    def prepare_folder(self):
        """Prepare a single folder to store all images. 
        """
        if not os.path.exists(self.args.img_dir):
            os.makedirs(self.args.save_dir)

        for index in self.imageinfo_by_id.keys():
            url = self.imageinfo_by_id[index][idx_URL]
            # print(url)
            # response = requests.get(url)
            # img = Image.open(BytesIO(response.content))
            # img.save(save_dir+"{:s}.jpg".format(index))
            img_path = save_dir+"{:s}.jpg".format(index)
            if fetch_and_save_image(img_path, url):
                self.remove_broken_path(img_path)
        print("Number of broken images: " + str(len(self.brokenImages)))


    # def saveJsonFile(self, save_dir):
    #     """Save three json files to save_dir
    #     """
    #     if save_dir[-1] != "/":
    #         save_dir += "/"
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     str_visual_concepts = json.dumps(self.visual_concepts)
    #     str_imageautotag_by_id = json.dumps(self.imageautotag_by_id)
    #     str_imageinfo_by_id = json.dumps(self.imageinfo_by_id)

    #     with open(save_dir+"visual_concepts.json", "a") as f:
    #         f.write(str_visual_concepts)

    #     with open(save_dir+"autotags_by_id.json", "a") as f:
    #         f.write(str_imageautotag_by_id)

    #     with open(save_dir+"imageinfo_by_id.json", "a") as f:
    #         f.write(str_imageinfo_by_id)

    # def saveImgByConcepts(self, save_dir, min_score=0):
    #     """Save all images by visual concepts into save_dir
    #         Only image score above min_score will be stored into dataset
    #     """
    #     concept_dir = save_dir+"concepts/"
    #     if not os.path.exists(concept_dir):
    #         os.makedirs(concept_dir)

    #     for concept in self.visual_concepts:
    #         list_of_idx = self.visual_concepts[concept]
    #         for (id_img, score) in list_of_idx:
    #             if float(score) >= min_score:
    #                 # Save image
    #                 this_concept_dir = concept_dir + concept + "/"
    #                 if not os.path.exists(this_concept_dir):
    #                     os.makedirs(this_concept_dir)
                    
    #                 url = self.imageinfo_by_id[id_img][idx_URL]
    #                 img_path = this_concept_dir+"{:s}.jpg".format(id_img)
    #                 if fetch_and_save_image(img_path, url):
    #                     self.remove_broken_path(img_path)
    #     print("Number of broken images: " + str(len(self.brokenImages)))



if __name__ == "__main__":
    args = parser.parse_args()
    parser = FlickrParser(args)
    parser.prepare_folder()
    parser.load_files()
    # parser.matchWithAutotagFile(autotags_path)

    # autotags_path = "/phoenix/S2/zhiqiu/depth_learning/yfcc100m_autotags"
    # if args.original_size:
    #     save_dir = "/phoenix/S3/zhiqiu/flickr_unique_{:d}_original/".format(max_images)
    # else:
    #     save_dir = "/phoenix/S3/zhiqiu/flickr_unique_{:d}_medium/".format(max_images)
    
    # if args.task == "dataset":
    #     datasets_path = "/phoenix/S2/ukm4/flickr/yfcc100m_dataset"
    #     parser.readDatasetFile(datasets_path)
    # elif args.task == "noah":
    #     datasets_path = "/phoenix/S2/snavely/data/Flickr100M/flickr100m_data/imageid_user_mediumurl_origurl.txt"
    #     parser.readNoahFile(datasets_path)
    # # max_images = 2
    # # max_images = 2
    
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # img_dir = save_dir + "images/"
    # json_dir = save_dir + "json/"
    
    # parser.saveJsonFile(json_dir)
    # parser.saveImgByConcepts(save_dir)