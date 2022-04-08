from deepface import DeepFace
import logging
import os
import shutil


if __name__ == '__main__':
    os.mkdir('../data/race_sorted_images/')

    for dirpath, dirnames, files in os.walk('../cropped/'):
        dirname = dirpath.split('/')[-1]
        if dirname:
            sorted_directory_path = '../data/race_sorted_images/{}/'.format(dirname)
            os.mkdir(sorted_directory_path)

            for filename in files:
                try:
                    print(filename)
                    full_path = '{}/{}'.format(dirpath, filename)
                    obj = DeepFace.analyze(img_path=full_path, actions=['race'], enforce_detection=False)
                    races = obj['race']
                    most_likely_race = max(races, key=races.get)
                    print(most_likely_race)

                    race_directory = '{}/{}/'.format(sorted_directory_path, most_likely_race)

                    try:
                        os.mkdir(race_directory)
                    except FileExistsError as exc:
                        pass

                    shutil.copy(full_path, race_directory)
                except Exception as e:
                    logging.info(e)
