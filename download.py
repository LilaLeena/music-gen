import requests
import pickle

tunes = []
batch_size = 1000
for batch in range(0, 20):
  for tune in range(0, batch_size):
    tune = batch*batch_size+tune
    url = 'https://thesession.org/tunes/%d/abc/200000' % tune
    r = requests.get(url)
    if r.status_code != 200:
      continue
    else:
      splits = ['X:' + split for split in r.text.split('X:') if len(split)>10]
      tunes.append(splits)
      if len(tunes) % 100 == 0:
        print('Downloaded %d tunes' % len(tunes))
  
  with open('data.pickle', 'wb') as f:
    pickle.dump(tunes, f)

with open('data.pickle', 'wb') as f:
  pickle.dump(tunes, f)
    
