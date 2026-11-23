# 🚢 Część 1: Titanic — Kurs Klasyfikacji od Podstaw

Ten moduł to pierwsza część mojego praktycznego przewodnika po Data Science i Machine Learningu, który tworzyłem podczas własnej nauki Pythona od podstaw.

Na przykładzie historycznych danych o pasażerach **Titanica** przechodzimy krok po kroku przez cały problem **klasyfikacji binarnej** — czyli uczenia komputera, jak na podstawie cech pasażera (płci, wieku, klasy biletu, wielkości rodziny czy tytułu grzecznościowego) przewidzieć jedną z dwóch kategorii: **`1` (Przeżył)** albo **`0` (Nie przeżył)**.

Nie znajdziesz tu przekombinowanych sztuczek pod wyśrubowanie wyniku w rankingu. Celem tego kursu jest **spokojne zrozumienie całego procesu**, zobaczenie każdego kroku na wykresach i wygenerowanie czystego pliku startowego (`submission_titanic_baseline.csv`), od którego możesz zacząć własne eksperymenty na Kaggle.

---

## 🗂️ Spis lekcji (`notebooks/`)

1. **`01_poznajemy_dane_i_klasyfikacje.ipynb`**
   * Czym różni się klasyfikacja (kategoria `0`/`1`) od regresji (ciągła liczba).
   * Rola plików `train.csv` (891 pasażerów), `test.csv` (418 pasażerów) i `gender_submission.csv`.
   * Dlaczego średnia z kolumny zero-jedynkowej (`.mean()`) od razu daje procent ocalałych (`38.4%`) oraz jak wyglądają rozkłady wieku (`Age`) i cen biletów (`Fare`).
2. **`02_kto_przezyl_wizualizacja_zaleznosci.ipynb`**
   * Jak działa grupowanie danych `.groupby()` (*Rozdziel $\rightarrow$ Policz $\rightarrow$ Połącz*).
   * Wizualizacja zasady *„Kobiety i dzieci przodem”* (`74.2%` ocalałych kobiet vs `18.9%` mężczyzn) oraz schodów klasowych (`63.0%` w 1. klasie vs `24.2%` w 3. klasie).
3. **`03_brakujace_dane_i_kodowanie_tekstu.ipynb`**
   * Diagnoza braków `NaN` w obu tabelach.
   * Jak zamienić 77% brakujących numerów kabin (`Cabin`) w potężną cechę zero-jedynkową **`Has_Cabin`** (`66.7%` przeżycia z kabiną vs `30.0%` bez kabiny).
   * Uzupełnianie liczb medianą (`.median()`), tekstu dominantą (`.mode()[0]`) oraz kodowanie słów na liczby (`.map()`, `pd.get_dummies()`).
4. **`04_pierwszy_model_i_macierz_pomylek.ipynb`**
   * Podział danych na zbiór treningowy i „sejf walidacyjny” (`train_test_split` z `random_state=42`).
   * Pierwsze Drzewo Decyzyjne (`DecisionTreeClassifier`), miara dokładności **Accuracy (`accuracy_score`)**, zjawisko przeuczenia (*Overfitting*) i ograniczenie głębokości drzewa (`max_depth=4`).
   * Jak czytać 4 kratki **Macierzy Pomyłek (*Confusion Matrix*)**: `TN`, `TP`, `FP`, `FN`.
5. **`05_inzynieria_cech_tytuly_i_rodziny.ipynb`**
   * Tworzenie cech rodzinnych (`FamilySize = SibSp + Parch + 1` oraz `IsAlone`) — dlaczego najlepiej radziły sobie rodziny 2–4 osobowe.
   * Wyciąganie tytułu (`Title`) z kolumny `Name` — odkrycie grupy 40 małych chłopców z tytułem **`Master`** (mediana wieku `3.5 roku`) oraz inteligentne uzupełnianie brakującego wieku medianą właściwą dla każdego tytułu.
6. **`06_walidacja_krzyzowa_i_porownanie_modeli.ipynb`**
   * Dlaczego w klasyfikacji używamy walidacji warstwowej **`StratifiedKFold(n_splits=5)`**.
   * Jak działa **Regresja Logistyczna (`LogisticRegression`)** i dlaczego mimo nazwy służy do klasyfikacji.
   * Turniej 4 klasyfikatorów (Drzewo Decyzyjne, Regresja Logistyczna, Las Losowy, Gradient Boosting) oraz sprawdzenie prawdopodobieństw przez `.predict_proba()`.
7. **`07_finalny_model_i_submission_baseline.ipynb`**
   * Czysty potok przetwarzania `train.csv` i `test.csv` bez wycieku danych (*Data Leakage*).
   * Trening zwycięskiego modelu na 100% danych treningowych, lista kontrolna (*Sanity Check*) i wygenerowanie pliku **`../submission_titanic_baseline.csv`**.
   * **Wielka Ściąga Data Scientista:** tabela podsumowująca wszystkie różnice między Klasyfikacją (Titanic) a Regresją (House Prices).

---

## 🛠️ Jak uruchomić ten moduł?

1. Pobierz pliki danych z konkursu [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) (`train.csv`, `test.csv`, `gender_submission.csv`) i umieść je w katalogu:
   ```text
   01_titanic_klasyfikacja/data/
   ```
2. Upewnij się, że masz zainstalowane biblioteki z głównego pliku `requirements.txt` (`pip install -r ../requirements.txt`).
3. Otwieraj kolejne notatniki w folderze `notebooks/` od `01` do `07` i uruchamiaj komórki skrótem `Shift + Enter`.
