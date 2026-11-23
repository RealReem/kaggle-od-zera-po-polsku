# 🏠 Część 2: House Prices — Kurs Regresji od Podstaw

Ten moduł to druga część mojego praktycznego kursu Data Science i Machine Learningu, który powstał podczas mojej nauki Pythona od podstaw.

O ile w Części 1 (Titanic) uczyliśmy komputer przewidywać kategorię (`0` lub `1`), o tyle tutaj wchodzimy w świat **regresji** — czyli przewidywania **konkretnej liczby ciągłej**: ceny sprzedaży domu w dolarach (**`SalePrice`**) na podstawie 80 kolumn opisujących metraż, jakość wykończenia, garaż, piwnicę, rok budowy i dzielnicę.

Cały kurs został napisany prostym, obrazowym językiem („dla totalnego laika”), z dokładnym wyjaśnieniem każdego nawiasu w kodzie oraz intuicji matematycznej stojącej za logarytmami, kwartylami, korelacją i modelami drzewiastymi. Zamiast gonić za sztucznym wynikiem na tablicy Kaggle, krok po kroku budujemy solidny, zrozumiały model bazowy i generujemy gotowy plik startowy `submission` pod dalsze własne eksperymenty.

---

## 🗂️ Spis lekcji (`notebooks/`)

1. **`01_poznajemy_dane.ipynb`**
   * Po co są 4 pliki w folderze `data/` (`train.csv`, `test.csv`, `data_description.txt`, `sample_submission.csv`).
   * Pierwsze kroki w `pandas`: `.read_csv()`, `.shape`, `.head()`, `.describe()` oraz różnica między nawiasami okrągłymi `()` a kwadratowymi `[]`.
   * Histogram cen `SalePrice` i wyjaśnienie na przykładzie „miliardera w barze”, dlaczego średnia (`180 921 $`) jest wyraźnie wyższa od mediany (`163 000 $`).
2. **`02_wizualizacja_i_zaleznosci.ipynb`**
   * Szczegółowe wyjaśnienie logarytmu (`np.log1p` oraz `np.expm1`): dlaczego logarytm zamienia skośny wykres cen w symetryczny „dzwon” i pozwala modelowi myśleć procentowo.
   * Wykres punktowy (`plt.scatter`) dla metrażu `GrLivArea` oraz wykrycie 2 gigantycznych anomalii (*Outlierów*).
   * Matematyczna anatomia wykresu pudełkowego (`.boxplot()`) dla oceny jakości `OverallQual` (kwartyle $Q_1, Q_2, Q_3$, rozstęp $\text{IQR}$ oraz wąsy $1.5 \times \text{IQR}$).
3. **`03_czyszczenie_danych.ipynb`**
   * Dlaczego puste kratki (`NaN`) psują równania matematyczne modeli ML i jak pod maską działa łańcuch `.isnull().sum()` (`True = 1`, `False = 0`).
   * Śledztwo w `data_description.txt`: odróżnienie **pozornego braku danych** (gdzie skrót `NA` oznacza fizyczny brak basenu `PoolQC` lub garażu `GarageType` $\rightarrow$ uzupełniamy słowem `'Brak'` lub `0`) od **prawdziwej dziury w danych** (`LotFrontage` uzupełniane medianą, `Electrical` dominantą `.mode()[0]`).
4. **`04_pierwszy_model.ipynb`**
   * Jak myśli pojedyncze Drzewo Decyzyjne, a jak Las Losowy (`RandomForestRegressor` — rada 100 rzeczoznawców).
   * Podział na zbiór treningowy i „sejf walidacyjny” (`train_test_split` z `random_state=42`), 3 kroki w `scikit-learn` (stworzenie $\rightarrow$ `.fit()` $\rightarrow$ `.predict()`) oraz wzór na Średni Błąd Bezwzględny (**MAE**).
   * Wykres *Prawdziwa cena vs Przewidziana cena* oraz śledztwo wokół 5 domów, na których nasz pierwszy model pomylił się najbardziej.
5. **`05_pierwszy_submission.ipynb`**
   * Dlaczego przed egzaminem trenujemy model na 100% danych z `train.csv`.
   * Pułapka nowych braków `NaN` w `test.csv` w kolumnach, które w `train.csv` były idealnie czyste.
   * Podsumowanie 3 rodzajów nawiasów w Pythonie (`()`, `[]`, `{}`), zapis pliku `.to_csv('submission.csv', index=False)` i wykres kontrolny rozkładu predykcji (*Sanity Check*).
6. **`06_cechy_tekstowe_i_encoding.ipynb`**
   * Dwie rodziny cech tekstowych: cechy **porządkowe (*Ordinal*)** z naturalną drabiną jakości (`KitchenQual`, `ExterQual`) kontra cechy **nominalne (*Nominal*)** bez drabiny (`Neighborhood`).
   * Kodowanie porządkowe słownikiem `.map()` oraz kodowanie zero-jedynkowe (*One-Hot Encoding*) przez `pd.get_dummies(..., dtype=int)` wraz z wykresem spadku błędu MAE.
7. **`07_walidacja_krzyzowa_kfold.ipynb`**
   * Dlaczego pojedynczy `train_test_split` przypomina loterię i jak działa 5-krotna walidacja krzyżowa (`KFold(n_splits=5, shuffle=True, random_state=42)`).
   * Jak Fold nr 3 ujawnił niszczący wpływ dwóch anomalii z Lekcji 2 (`GrLivArea > 4000`) i jak po ich usunięciu średni błąd spada o ponad 1 000 \$.
8. **`08_nowe_cechy_feature_engineering.ipynb`**
   * Dlaczego drzewa decyzyjne nie potrafią same dodawać kolumn i jak tworzymy 4 nowe cechy: `TotalSF` (całkowity metraż budynku), `TotalBath` (łączna liczba łazienek), `HouseAge` (wiek domu) oraz `RemodAge` (lata od remontu).
   * Matematyka współczynnika korelacji Pearsona (`.corr()` od `-1.0` do `+1.0`) oraz wykres ważności cech (`feature_importances_`), na którym `TotalSF` zajmuje 1. miejsce.
9. **`09_porownanie_modeli_i_podsumowanie.ipynb`**
   * Wielki Turniej 3 Modeli w 5-Fold CV: Pojedyncze Drzewo Decyzyjne (`24 594 $`) vs Las Losowy (`16 960 $`) vs **Gradient Boosting (`16 120 $`)**.
   * Jak krok po kroku działa sztafeta *Gradient Boosting*, co oznaczają hiperparametry `n_estimators`, `learning_rate` i `max_depth` oraz **7-punktowa Check-lista na Twój kolejny samodzielny konkurs Kaggle**.

---

## 🛠️ Jak uruchomić ten moduł?

1. Pobierz pliki danych z konkursu [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (`train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt`) i umieść je w katalogu:
   ```text
   02_house_prices_regresja/data/
   ```
2. Upewnij się, że masz zainstalowane biblioteki z głównego pliku `requirements.txt` (`pip install -r ../requirements.txt`).
3. Otwieraj kolejne notatniki w folderze `notebooks/` od `01` do `09` i uruchamiaj komórki skrótem `Shift + Enter`.
