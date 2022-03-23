from google_images_search import GoogleImagesSearch
import os
import json
import requests


class ImageDownloader(GoogleImagesSearch):
    """
    Extended GoogleImagesSearch class from the google_image_search library
    """

    def __init__(self, developer_key, custom_search_cx,
                 progressbar_fn=None, validate_images=True):
        super().__init__(developer_key, custom_search_cx,
                 progressbar_fn, validate_images)
        # Our failed attempt to make it download the next 100 images (10 pages, 10 images each for pages 0-9):
        # self._page = 10



def my_progressbar(url, progress):
    """ Too verbose, ended up not using it. """
    print(url + ' ' + str(progress) + '%')

def download_images(gis):
    i = 1
    for image in gis.results():
        print(i)
        image.referrer_url
        image.download('/Users/Eo/Documents/Studies/COMP6721/project/tmp/')
        i += 1


def main(search_term = 'n95 mask'):
    """ Searches Google Images for the passed search terms filtering out
    the results by face images only.
    :param search_term:
    :return:
    """

    # Google Cloud project credentials used for the project
    gis = ImageDownloader('AIzaSyBmVTUi5qeFXiEH9x8lt0wmr_Wu9xfg8n0', 'dba8a0b5c9d9fa54e')

    _search_params = {
        'q': search_term,
        'num': 100,
        'ijn': 2,
        'safe': 'medium',  ##
        'imgType': 'face',  ##
        'imgColorType': 'color'  ##
    }

    gis.search(search_params=_search_params,
               path_to_dir='/Users/Eo/Documents/Studies/COMP6721/project/tmp/')

    download_images(gis)
    gis.next_page()


def download_images_from_json(path = '/Users/Eo/Documents/Studies/COMP6721/project/n95_valve_json'):
    """
    Parses JSON files generated via SerpAPI's free tier Google Images search:
    https://serpapi.com/images-results. Queries used included "person wearing n95 mask",
    "wearing n95 mask with valve", and others. The JSONs can be found under the directory "jsons"
    :param path:
    :return:
    """

    os.chdir(path)
    with open(path + '/n95_valve.json', 'rb') as file:
        links = json.load(file)
        for obj in links['images_results']:
            url = obj.get('original')
            if not url:
                continue
            print(url)
            filename = url.split('/')[-1]
            if filename[-4:] not in ['jpeg', '.jpg', '.png']:
                filename += '.jpeg'

            try:
                img_data = requests.get(url, timeout=7).content
                with open(filename, 'wb') as handler:
                    print('Saving as ' + filename)
                    handler.write(img_data)
            except requests.exceptions.ReadTimeout:
                continue
            except Exception:
                continue

if __name__=='__main__':
    # Parameters were updated on a case by case basis
    main()
    download_images_from_json()

