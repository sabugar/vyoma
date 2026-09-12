import json, sys, urllib.request, logging
sys.path.insert(0,'/home/ubuntu/pocket-infer-sw/python')
logging.disable(logging.CRITICAL)
from pocketinfer.applications.hear_the_world import HearTheWorld as H
def post(u,p,t=300):
    r=urllib.request.Request(u,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=t) as x: return json.load(x)
B="http://127.0.0.1:11400"
# (topic, hindi question, key clinical words that must survive)
Q=[
("feeding","बच्चे को कब तक सिर्फ माँ का दूध देना चाहिए",["दूध"]),
("feeding","छह महीने के बाद बच्चे को क्या खिलाना चाहिए",["महीने"]),
("anaemia","बच्चे में खून की कमी के क्या लक्षण हैं",["खून","लक्षण"]),
("malnutrition","बच्चे का वजन उम्र के हिसाब से कम है क्या करें",["वजन"]),
("malnutrition","कुपोषित बच्चे को कब रेफर करना चाहिए",["कुपोषित"]),
("immunisation","खसरे का टीका कब लगता है",["खसरे","टीका"]),
("immunisation","बच्चे को कौन कौन से टीके लगवाने हैं",["टीके"]),
("sickchild","बीमार बच्चे में खतरे के लक्षण क्या हैं",["खतरे","लक्षण"]),
("fever","बच्चे को तेज बुखार है क्या दवा दें",["बुखार","दवा"]),
("fever","बुखार के साथ गर्दन अकड़ जाए तो क्या करें",["गर्दन"]),
("diarrhoea","बच्चे को दस्त लग गए हैं क्या करें",["दस्त"]),
("diarrhoea","ओ आर एस कैसे बनाते हैं",["आर","एस"]),
("diarrhoea","दस्त में पानी की कमी के लक्षण क्या हैं",["पानी","कमी"]),
("ari","बच्चे की साँस तेज चल रही है तो क्या करें",["साँस"]),
("ari","पसली चलना किस बीमारी का लक्षण है",["पसली"]),
("abortion","गर्भपात कितने हफ्ते तक सुरक्षित है",["गर्भपात","हफ्ते"]),
("abortion","गर्भपात के बाद खतरे के लक्षण क्या हैं",["गर्भपात"]),
("familyplanning","परिवार नियोजन के कौन से साधन हैं",["परिवार","नियोजन"]),
("familyplanning","कॉपर टी किसे नहीं लगवानी चाहिए",["कॉपर"]),
("familyplanning","गर्भनिरोधक गोली किसे नहीं देनी चाहिए",["गोली"]),
("rti","महिला को सफेद पानी की शिकायत है क्या करें",["सफेद","पानी"]),
("lbw","कम वजन के नवजात को कैसे गर्म रखें",["नवजात","वजन"]),
("lbw","कम वजन के बच्चे को दूध कैसे पिलाएं",["दूध"]),
("asphyxia","बच्चा जन्म के बाद रो नहीं रहा तो क्या करें",["जन्म"]),
("asphyxia","मयूकस एक्सट्रैक्टर का उपयोग कैसे करें",["एक्सट्रैक्टर"]),
("sepsis","नवजात में सेप्सिस के लक्षण क्या हैं",["सेप्सिस","लक्षण"]),
("sepsis","बच्चे की नाभि लाल है और मवाद आ रहा है",["नाभि","मवाद"]),
("malaria","मलेरिया की जांच कैसे करते हैं",["मलेरिया","जांच"]),
("malaria","मलेरिया में कौन सी दवा देनी है",["मलेरिया","दवा"]),
("tb","टीबी के मरीज को क्या सलाह दें",["टीबी","मरीज"]),
]
rows=[]
for topic,hi,keys in Q:
    try:
        tts=post(B+"/tts",{"text":hi,"language":"hi"})
        asr=post(B+"/asr",{"audio_base64":tts["audio_base64"],"language":"hi"})
        heard=H._normalise_transcript(asr.get("text","").strip())
    except Exception as e:
        heard="<error %s>" % e
    lost=[k for k in keys if k not in heard]
    rows.append(dict(topic=topic,hi=hi,heard=heard,keys=keys,lost=lost))
json.dump(rows,open('/tmp/claude-1000/-home-ubuntu/7b59ed87-9f09-400a-b3d9-1cd169fe5ddc/scratchpad/asrprobe.json','w'),ensure_ascii=False,indent=1)
ok=sum(1 for r in rows if not r['lost'])
print("   survived: %d/%d" % (ok,len(rows)))
