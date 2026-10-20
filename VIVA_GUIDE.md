# 🎓 Viva Guide — Sir ko har cheez asan alfaaz mein samjhane ke liye

Ye guide tumhare liye hai. Har file, har important line, aur sir ke
mumkin sawalon ke jawab — sab simple alfaaz mein.

---

## 1. Project ek line mein

> "Sir, ye ek full-stack ML web app hai. Ek model flight ki arrival
> delay predict karta hai (regression), doosra model passenger ki
> satisfaction predict karta hai (classification). Backend FastAPI
> hai, frontend ek HTML page hai."

## 2. Dataset

- Kaggle se liya: **Airline Passenger Satisfaction**
- Total **129,880 rows**: train.csv mein 103,904 aur test.csv mein 25,976
  (split dataset ke saath hi aya tha, mene khud nahi banaya)
- Har row = ek passenger: uski age, class, flight distance, delays,
  14 services ki ratings (0–5), aur aakhir mein `satisfaction` label.

## 3. Model 1 — Linear Regression (Delay wala)

**Sawal: Regression kya hota hai?**
> "Jab answer ek number ho (price, delay, marks) to wo regression hai.
> Jab answer ek category ho to wo classification hai."

**Mera model kya karta hai?**
- Input: `Departure Delay` + `Flight Distance`
- Output: `Arrival Delay` (minutes mein)
- Formula jo model ne seekha: `Arrival Delay ≈ 0.92 + 0.98 × Departure Delay`
  Matlab flight jitni late niklegi, takreeban utni hi late pohchegi.

**Sawal: R² = 0.9302 ka matlab?**
> "R² batata hai model data ko kitna achi tarah samajhta hai. 1.0
> perfect hota hai. Mera 0.93 hai, matlab model arrival delay ka
> 93% pattern samajh gaya. MAE 5.27 ka matlab hai ke average
> prediction sirf 5.27 minutes se ghalat hoti hai."

**Code ki main lines (`train_models.py`):**
- `dropna()` — jin rows mein arrival delay missing tha, wo hata diye.
- `LinearRegression()` — model banaya.
- `.fit(X_train, y_train)` — model ne train data se seekha.
- `.predict(X_test)` — test data par predictions kin.
- `r2_score()` — predictions ko asli jawabon se compare karke score.
- `joblib.dump()` — trained model ko file mein save kar diya, taake
  bar bar train na karna pare.

## 4. Model 2 — Logistic Regression (Satisfaction wala)

**Sawal: Naam mein "Regression" hai, lekin ye classification kyun hai?**
> "Sir, Logistic Regression andar probability calculate karta hai.
> Agar probability 0.5 se zyada ho to class 1 (satisfied), warna
> class 0. Is liye ye category predict karta hai, is liye ye
> classification model hai."

**Input:** 14 service ratings + Age + Flight Distance + 4 text wali
cheezein (Gender, Customer Type, Type of Travel, Class)

**Do mushkil cheezein aur unka hal:**
1. **Text data:** Model words nahi parh sakta. `OneHotEncoder` har word
   ko 0/1 walay columns mein badal deta hai. Misal: Class ke 3 options
   (Eco, Eco Plus, Business) 3 columns ban jate hain, sahi wale mein 1.
2. **Numbers ke mukhtalif sizes:** Age 7–85 hai, ratings 0–5, distance
   hazaaron mein. `StandardScaler` sab numbers ko ek jaisi scale par
   le ata hai taake koi ek number doosron par haavi na ho.

**Pipeline kya hai?**
> "Pipeline matlab steps ko ek line mein jor dena: pehle preprocessing
> (scaler + encoder), phir model. Data khud ba khud dono steps se
> guzar kar prediction deta hai. Ek hi cheez train bhi hoti hai aur
> save bhi."

**Scores:**
- **Accuracy = 87.04%** — 100 test passengers mein se 87 ka sahi jawab.
  (Sirf sab ko "dissatisfied" kehne wala andaza 56% hota — model us se
  bohat behtar hai.)
- **ROC-AUC = 0.9248** — model satisfied aur dissatisfied passengers
  ko alag karne mein kitna acha hai; 1.0 perfect, 0.5 andaza hota hai.
- `predict()` category deta hai, `predict_proba()` confidence deta hai
  (masalan 0.99 matlab 99% yaqeen ke passenger satisfied hai).

**Answer ko 0/1 kyun banaya?**
> "`(satisfaction == 'satisfied').astype(int)` — satisfied = 1,
> baqi = 0. Model numbers se seekhta hai, words se nahi."

## 5. Backend — FastAPI (`app/main.py`)

**Sawal: FastAPI kya kar raha hai?**
> "FastAPI Python ka server hai. Wo dono saved models ko `joblib.load`
> se load karta hai, aur do raste (endpoints) deta hai jahan website
> data bhej kar prediction le sakti hai."

**Do required endpoints:**
- `POST /predict/regression` — JSON mein 2 numbers ate hain, model se
  prediction le kar JSON mein jawab.
- `POST /predict/classification` — JSON mein passenger details + ratings
  ate hain, jawab: category + probability.

**Baaki important lines:**
- `BaseModel` (Pydantic) classes input ki **shape aur rules** define
  karti hain. Rating 5 se zyada bhejo to FastAPI khud **422 error**
  de deta hai — ghalat data model tak jata hi nahi.
- Model ko wohi **column names** chahiye jo training mein thay, is liye
  endpoint ke andar input ko ek choti pandas table (DataFrame) mein
  badla jata hai.
- Models **server start hote hi ek dafa** load hote hain, har request
  par nahi — is liye jawab foran milta hai.
- `GET /` website (HTML page) deta hai, `GET /model-info` scores deta
  hai jo page ke header mein nazar ate hain.

**Sawal: POST hi kyun, GET kyun nahi?**
> "GET data lene ke liye hota hai, POST data bhejne ke liye. Hum
> server ko input bhej rahe hain, is liye POST."

**Sawal: JSON kya hai?**
> "JSON do programs ke darmiyan data bhejne ka simple text format
> hai — `{ "key": value }` ki shakal mein."

## 6. Frontend — HTML page (`app/static/index.html`)

- **Ek hi page**, us par **do alag forms** — ek har model ke liye
  (bilkul assignment ki requirement).
- CSS page ko khoobsurat, responsive (mobile par ek column) aur
  animated banata hai. Koi library nahi, sab apna CSS hai.
- JavaScript ke 4 kaam:
  1. `/model-info` se scores la kar header mein dikhana.
  2. Sliders ki value sath sath dikhana (ratings star ★ mein bhi).
  3. Preset buttons se form ek click mein bharna (demo ke liye).
  4. Form submit par `fetch()` se backend ko JSON bhejna aur jawab
     ko animated number/bar ke sath dikhana.
- `event.preventDefault()` page ko reload hone se rokta hai, warna
  prediction nazar hi na aye.
- `async/await` ka matlab: server ke jawab ka intezar karo, page ko
  freeze mat karo.

## 7. Project chalane ke steps (agar sir poochein / demo ho)

```bash
pip install -r requirements.txt   # libraries
python train_models.py            # models train + save (scores print honge)
uvicorn app.main:app --reload     # server start
```
Phir browser mein: **http://127.0.0.1:8000**
API ki automatic documentation: **http://127.0.0.1:8000/docs**

**Demo ka best tareeqa:** pehle *Heavily delayed* preset → Predict,
phir *Happy passenger* preset → Predict. Dono results 10 second mein.

## 8. Sir ke mushkil sawal — tayyar jawab

**Q: Train/test split kyun alag rakha?**
> "Jo data model ne dekha hi na ho (test), usi par asli imtihan hota
> hai. Warna model ratta laga kar bhi full marks le lega — isay
> overfitting kehte hain."

**Q: Logistic Regression hi kyun, koi aur model kyun nahi?**
> "Assignment mein classification chahiye tha, aur Logistic Regression
> sab se simple aur samjhane mein asan classification model hai, aur
> ye probability bhi deta hai."

**Q: Delay model ka R² itna high kyun hai? Koi leakage to nahi?**
> "Departure delay flight ke nikalne ke baad maloom hota hai, ye
> future ki maloomat nahi hai. Flight late nikli ho to raste mein
> time pura karna mushkil hota hai, is liye relation naturally
> strong hai. Haan, ye model flight se dinon pehle ki forecast
> nahi hai — ye limitation mene README mein khud likhi hai."

**Q: Satisfaction model ki limitation?**
> "Ratings passenger khud survey mein deta hai, to ye model ratings
> milne ke baad satisfaction samajhne ke liye hai — airline ko ye
> pata chalta hai ke kaunsi services satisfaction ko sab se zyada
> affect karti hain."

**Q: joblib kyun use kiya?**
> "Trained model ko file mein save karne ke liye, taake website har
> dafa model ko dobara train na kare. `joblib.dump` save karta hai,
> `joblib.load` wapas load karta hai."

**Q: Agar naya passenger aye jiski Class koi nayi value ho?**
> "`OneHotEncoder(handle_unknown='ignore')` rakha hai — nayi value
> par error nahi aye ga, wo columns 0 reh jayenge."

## 9. Ek saans mein pura flow

```
Kaggle data → train_models.py → 2 models train → joblib se .pkl files
     → FastAPI un models ko load karta hai
     → User website par form bharta hai
     → JavaScript JSON bana kar POST karta hai
     → FastAPI model se prediction le kar JSON wapas bhejta hai
     → Website result animated andaaz mein dikha deti hai
```
