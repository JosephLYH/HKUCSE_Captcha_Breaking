import os
import requests

SAVEPATH = 'data_raw'

url = 'https://bs.cse.hku.hk/ihpbooking/stickyImg'
num_samples = 50

if not os.path.exists(SAVEPATH):
    os.mkdir(SAVEPATH)

for i in range(num_samples):
    response = requests.get(url)

    # # save as jpg
    # with open(SAVEPATH + str(i) + '.jpg', 'wb') as f:
    #     f.write(response.content)

    # save as png
    with open(os.path.join(SAVEPATH, str(i) + '.png'), 'wb') as f:
        f.write(response.content)