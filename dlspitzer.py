import csv
import pathlib
import urllib.request


def get_spitzer_url(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data[1:]

def dl_spitzer(data):
    for l in data:
        swift, url = l
        print(l)
        swift_dir = pathlib.Path('spitzer/' + swift + '/spitzer')
        swift_dir.mkdir(parents=True, exist_ok=True)
        save_name = swift_dir.joinpath(url.split('/')[-1])
        if url != 'null':
            print(save_name)
            urllib.request.urlretrieve(url, save_name)

if __name__ == '__main__':
    data = get_spitzer_url("bass_irsa.csv")
    dl_spitzer(data)
