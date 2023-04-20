import requests

res_ = requests.get("http://checkip.amazonaws.com/")

print("Your IP", res_.text)
import urllib.request
from urllib.parse import quote
import httplib2
import json
import os

# ①GCPから取得したAPI
API_KEY = "AIzaSyAf0_Ub12QvytIiOvXFsZomLYRHiqhgJ8g"

# ②GoogleのCustom Search EngineのID
CUSTOM_SEARCH_ENGINE = "10072a841ab644cee"

# 取得したい画像の検索キーワード
keywords = ["小島 よしお"]
# urlを入れるリスト
img_list = []


# urlを取得する関数
def get_image_url(keyword, num):
    # num_searched = 検索した数 num = 検索したい数
    num_searched = 0
    while num_searched < num:
        # numの部分の例（95枚取得したい場合、90枚までは10枚ずつ取得　し残りが5枚）※検索結果は１ページあたり１０個の画像が変えるため
        query_img = "https://www.googleapis.com/customsearch/v1?key=" + API_KEY + "&cx=" + CUSTOM_SEARCH_ENGINE + "&num=" + str(10 if(num-num_searched)>10 else (num-num_searched)) + "&start=" + str(num_searched+1) + "&q=" + quote(keyword) + "&searchType=image"
        # 取得したurlを開く
        print(query_img)
        res = urllib.request.urlopen(query_img)
        data = json.loads(res.read().decode('utf-8'))
        #　jsonのitems以下のlinkに画像のurlが返ってくるのでそれを取り出しリストに追加
        for i in range(len(data["items"])):
            img_list.append(data['items'][i]['link'])
        # １ページあたり検索結果が10返ってくるので検索した数は10ずつ増やす
        num_searched += 10
    return img_list


# urlから画像を取得する関数
def get_image(keyword, img_list):
    # opener = urllib.request.build_opener()
    http = httplib2.Http(".cache")
    for i in range(len(img_list)):
        try:
            print(img_list[i])
            response, content = http.request(img_list[i])
            # ファイルパスを「origin_image/キーワード_index.jpg」とする
            filename = os.path.join("origin_image", keyword + "_" + str(i) + ".jpg")
            with open(filename, 'wb') as f:
                f.write(content)
        except:
            print("failed to download the image.")
            continue
            

# キーワードごとにurlを取得、その後urlから画像を抽出し指定したファイルに書き込む
for i in range(len(keywords)): 
    img_list = get_image_url(keywords[i], 200)
    get_image(keywords[i], img_list)  
    