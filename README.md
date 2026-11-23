# 🧭 Od Zera do Data Science: Praktyczny przewodnik po Kaggle (Titanic & House Prices)

Cześć! To repozytorium powstało podczas mojej własnej nauki Pythona, analizy danych i Machine Learningu od absolutnych podstaw.

Zaczynałem z czystej ciekawości i pasji do danych — chciałem zobaczyć na własne oczy, jak z surowych tabel wyciąga się ukryte zależności, jak komputer „uczy się” na przykładach i dlaczego jeden model myli się bardziej od drugiego. Tworzyłem to przede wszystkim dla siebie, jako uporządkowany podręcznik pisany ludzkim językiem, w którym każdy nawias, kropka w kodzie, wykres i wzór matematyczny są wytłumaczone krok po kroku.

W polskim internecie bardzo brakuje prostych, wizualnych materiałów, które przeprowadzają początkującego przez cały proces bez akademickiego żargonu i bez przeskakiwania trudnych momentów. Pomyślałem więc, że skoro te poradniki tak dobrze uporządkowały wiedzę u mnie, mogą stać się świetnym punktem startowym także dla innych osób stawiających pierwsze kroki w Data Science.

---

## 💡 Filozofia tego projektu: Zrozumienie zamiast wyścigu o ułamki procenta

Na Kaggle łatwo wpaść w pułapkę kopiowania cudzych, skomplikowanych skryptów tylko po to, by podbić wynik w rankingu o `0.001`. Tutaj cel jest zupełnie inny:
* **Pełne zrozumienie procesu od A do Z:** Od pierwszego otwarcia pliku `.csv`, przez czytanie wykresów, wykrywanie anomalii i mądre łatanie pustych komórek (`NaN`), aż po inżynierię cech (*Feature Engineering*) i uczciwą walidację krzyżową (*K-Fold Cross-Validation*).
* **Wszystko widać na wykresach:** Po każdym kroku rysujemy wykresy, żeby zobaczyć oczami, co się zmieniło i jak „myśli” model.
* **Gotowa baza startowa (*Baseline*):** Każdy z dwóch kursów kończy się czystym, wolnym od wycieków danych kodem, który generuje Twój pierwszy plik zgłoszeniowy (`submission`) na Kaggle. To solidny fundament, na którym możesz potem budować własne eksperymenty.

---

## 📚 Dwa moduły, które spinają się w jedną całość

Projekt składa się z dwóch osobnych, ale ściśle powiązanych ze sobą kursów w notatnikach Jupyter (`.ipynb`). Razem pokrywają dwa najważniejsze problemy klasycznego Machine Learningu:

### 1. 🚢 [Część 1: `01_titanic_klasyfikacja/`](./01_titanic_klasyfikacja/README.md) — Klasyfikacja (*Kto przeżył katastrofę Titanica?*)
Uczymy model podejmować decyzję zero-jedynkową: **`0` (Nie przeżył)** lub **`1` (Przeżył)**.
* **Spis 7 notatników w `01_titanic_klasyfikacja/notebooks/`:**
  1. `01_poznajemy_dane_i_klasyfikacje.ipynb` — Czym różni się klasyfikacja od regresji, słownik 12 kolumn i rozkład danych.
  2. `02_kto_przezyl_wizualizacja_zaleznosci.ipynb` — Zasada *„Kobiety i dzieci przodem”*, klasa biletu i grupowanie `.groupby()`.
  3. `03_brakujace_dane_i_kodowanie_tekstu.ipynb` — Łatanie braków `NaN`, zamiana brakujących kabin w cechę `Has_Cabin` i kodowanie tekstu.
  4. `04_pierwszy_model_i_macierz_pomylek.ipynb` — Pierwsze Drzewo Decyzyjne, sejf walidacyjny (`train_test_split`), przeuczenie (*Overfitting*) i Macierz Pomyłek (*Confusion Matrix*).
  5. `05_inzynieria_cech_tytuly_i_rodziny.ipynb` — Tworzenie nowych cech: wielkość rodziny (`FamilySize`, `IsAlone`), wyciąganie tytułów z nazwisk (`Title`, odkrycie chłopców `Master`) i mądre uzupełnianie wieku.
  6. `06_walidacja_krzyzowa_i_porownanie_modeli.ipynb` — Walidacja warstwowa `StratifiedKFold` oraz turniej 4 modeli (Drzewo, Regresja Logistyczna, Las Losowy, Gradient Boosting).
  7. `07_finalny_model_i_submission_baseline.ipynb` — Kompletny proces bez wycieku danych, wygenerowanie pliku `submission_titanic_baseline.csv` oraz **Wielka Ściąga porównująca Klasyfikację z Regresją**.

---

### 2. 🏠 [Część 2: `02_house_prices_regresja/`](./02_house_prices_regresja/README.md) — Regresja (*Ile kosztuje ten dom?*)
Uczymy model wyceniać nieruchomości, czyli przewidywać konkretną kwotę w dolarach (**`SalePrice`**) na podstawie 80 cech budynku i działki.
* **Spis 9 notatników w `02_house_prices_regresja/notebooks/`:**
  1. `01_poznajemy_dane.ipynb` — Rola 4 plików z Kaggle, pierwsze wczytanie tabel, wymiary `.shape` oraz zagadka średniej i mediany („miliarder w barze”).
  2. `02_wizualizacja_i_zaleznosci.ipynb` — Dlaczego logarytmujemy ceny (`np.log1p` i `np.expm1`), wykres punktowy metrażu `GrLivArea` (wykrycie 2 anomalii) oraz anatomia wykresu pudełkowego (`.boxplot()`) dla jakości `OverallQual`.
  3. `03_czyszczenie_danych.ipynb` — Jak działa `.isnull().sum()`, różnica między pozornym brakiem danych (`NA` = brak basenu/garażu w `data_description.txt`) a prawdziwą dziurą w danych (uzupełnianie medianą i dominantą `.mode()[0]`).
  4. `04_pierwszy_model.ipynb` — Jak działa Drzewo Decyzyjne i Las Losowy (`RandomForestRegressor`), podział `train_test_split`, średni błąd dolarowy `MAE` oraz śledztwo wokół 5 największych pomyłek modelu.
  5. `05_pierwszy_submission.ipynb` — Trening na 100% danych, pułapka nowych braków w `test.csv`, zapis `submission.csv` (`index=False`) i wykres kontrolny (*Sanity Check*).
  6. `06_cechy_tekstowe_i_encoding.ipynb` — Różnica między cechami porządkowymi (*Ordinal Encoding* przez `.map()`) a nieuporządkowanymi (*One-Hot Encoding* przez `pd.get_dummies()`).
  7. `07_walidacja_krzyzowa_kfold.ipynb` — Dlaczego pojedynczy podział danych potrafi oszukać i jak działa 5-krotna walidacja krzyżowa (`KFold`) oraz jak wpływa na nią usunięcie 2 anomalii z Lekcji 2.
  8. `08_nowe_cechy_feature_engineering.ipynb` — Tworzenie własnych cech (`TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`), matematyka korelacji Pearsona (`.corr()`) i ranking ważności cech (`feature_importances_`).
  9. `09_porownanie_modeli_i_podsumowanie.ipynb` — Wielki turniej modeli (Drzewo Decyzyjne vs Las Losowy vs Gradient Boosting), wyjaśnienie hiperparametrów oraz **7-krokowa Check-lista na kolejny konkurs Kaggle**.

---

## 🚀 Jak zacząć? (Instrukcja krok po kroku)

### 1. Zainstaluj wymagane biblioteki
W terminalu (w głównym folderze projektu) zainstaluj paczki z pliku `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Pobierz dane z Kaggle i umieść je w folderach `data/`
Ze względu na zasady licencyjne Kaggle surowe pliki `.csv` z danymi są wykluczone w pliku `.gitignore`. Pobierz je bezpośrednio ze stron obu darmowych konkursów:
* **Dla Części 1 (Titanic):** Pobierz pliki z [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) (`train.csv`, `test.csv`, `gender_submission.csv`) i wrzuć je do folderu:
  ```text
  01_titanic_klasyfikacja/data/
  ```
* **Dla Części 2 (House Prices):** Pobierz pliki z [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) (`train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt`) i wrzuć je do folderu:
  ```text
  02_house_prices_regresja/data/
  ```

### 3. Otwórz notatniki i uruchamiaj komórka po komórce
Otwórz folder w **VS Code** (z rozszerzeniem *Jupyter*) lub uruchom w terminalu `jupyter notebook`.
Każda lekcja kończy się krótkim zadaniem do samodzielnego wpisania (`# Miejsce na Twój kod:`) oraz podsumowaniem najważniejszych wniosków w 3 punktach.

---

## 📁 Struktura repozytorium

```text
.
├── README.md                              # Główny przewodnik po projekcie
├── requirements.txt                       # Lista bibliotek Pythona (pandas, numpy, matplotlib, scikit-learn)
├── .gitignore                             # Wykluczenie surowych plików CSV z data/ oraz prywatnych kopii roboczych
├── 01_titanic_klasyfikacja/               # CZĘŚĆ 1: Kurs Klasyfikacji (Titanic)
│   ├── README.md                          # Opis i instrukcja do kursu Titanica
│   ├── data/                              # Miejsce na pliki train.csv, test.csv, gender_submission.csv
│   └── notebooks/                         # Lekcje 01–07 (.ipynb)
└── 02_house_prices_regresja/              # CZĘŚĆ 2: Kurs Regresji (House Prices)
    ├── README.md                          # Opis i instrukcja do kursu House Prices
    ├── data/                              # Miejsce na pliki train.csv, test.csv, data_description.txt
    └── notebooks/                         # Lekcje 01–09 (.ipynb)
```

Powodzenia w odkrywaniu danych!
