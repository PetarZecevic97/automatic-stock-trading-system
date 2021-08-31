# Automatsko trgovanje na berzi

## Struktura projekta

- data: input podaci
- figures: grafovi rešenja
- KNN_prediction: grafovi knn predikcija za neke od parametara
- LR_figs: grafovi aproksimacija linearnim regresijama za neke od parametara
- notebooks: jupiter sveske
- results: rezultati modela

## Korišćene biblioteke

- numpy
- pandas
- sklearn

## Berza

Berze su institucije koje omogućavaju trgovinu hartija od vrednosti. Služe da spoje potražnju i ponudu po adekvatnoj ceni za obe strane, a za sebe uzimaju razliku u ceni (bid ask spread) kao proviziju. Cene akcija osciliraju tokom vremenskog perioda. Cena akcije je cena po kojoj se odvila poslednja transakcija. Trgovac može da kupi po toj ceni a može i da da nalog po nekoj svojoj ceni i da čeka da naiđe drugi trgovac kome ta cena odgovara. Trgovci koji daju naloge najviše utiču na cenu na berzi.

Trgovci imaju opciju da se nalaze u dve pozicije: long i short. Long pozicija znači da trgovac poseduje određeni broj akcija, dok short pozicija znači da je trgovac pozajmio određen broj akcija od neke institucije, prodao ih i sada ih duguje toj instituciji. U slučaju short pozicije trgovac je dužan da za ograničeno vreme vrati akcije sa nekom kamatom. Trgovac ulazi u short poziciju kada smatra da će cena opasti u doglednom periodu i želi da zaradi na tom padu. [[1](https://en.wikipedia.org/wiki/Stock_market)]

Automatsko trgovanje predstavlja vid trgovanja gde trgovac navede skup pravila po kojima se ulazi i izlazi iz pozicije tokom vremena, taj skup pravila isprogramira i ona mogu da se izvršavaju automatski preko računara. Danas je oko 80% trgovanja na berzi automatsko. [[2](https://www.academia.edu/28488586/Quantitative_Trading_Ernest_P_Chan)]

U ovom radu implementiran je i testiran originalni model za automatsko trgovanje nad podacima o cenama SPY indeksa.

## Podaci

Berzanski indeks predstavlja indikator stanja tržišta ili podskupa tržišta i služi trgovcima da mogu da uporede stanje tržišta kroz vreme. SPY (Standard and Poor's 500) indeks je berzanski indeks koji prati ponašanje 500 najvećih kompanija na berzi u SAD-u. Podaci su uzeti sa [[3](https://finance.yahoo.com/quote/SPY/history?p=SPY)].
Uzeti su podaci u periodu od početka 2000. godine do danas.
Podaci su vremenska serija sa pet kolona:

- open - cena sa kojom se otvara berza u konkretnom danu
- close - cena sa kojom se zatvara berza
- low - najniža cena u toku jednog dana
- high - najviša cena u toku jednog dana
- volume - ukupan broj akcija koje su učestovale u transakcijama u toku jednog dana

## Algoritam

Premisa problema je da imamo 100 000 dolara i na početku smo 2000. godine. Cilj nam je da maksimizujemo profit do poslednjeg dana koji imamo u podacima. Taksa na jednu transkaciju je 5% iznosa. Pod profitom se podrazumeva zbir preostalog kapitala i vrednosti svih akcija koje imamo u posedu u poslednjem danu. Kada biramo da trgujemo, mi u stvari biramo danas da ćemo trgovati sutra.

Algoritam pokušava da predvidi početak novog i kraj starog trenda i da u tim trenutcima trguje (za početak rastućeg trenda kupovina akcija, a za početak opadajućeg prodaja). Trendovi su opisani linearnim regresijama gde je cena funkcija vremena (radi se nad 'close' cenom). Za svaki sledeći dan se predviđa cena, model trenutnog trenda se ažurira sa tom predikcijom i posmatra se da li se greška modela povećala. Ako se povećala više od prosečne greške ranijih modela, algoritam predviđa kraj trenutnog trenda i početak novog i postavlja sledeći dan za početak nove linearne regresije.

Da bi se sračunala greška prvog modela, koristi se trening period u kom se ne trguje. Za taj period se naprave linearne regresije. Krajevi linearnih regresija se određuju presekom dugog i kratkog SMA-a cena. SMA (simple moving average) je srednja vrednost prethodnih n dana i računa se za svaki dan [[4](https://www.investopedia.com/terms/s/sma.asp#:~:text=A%20simple%20moving%20average%20is%20a%20technical%20indicator,simple%20moving%20average%20is%20used%20to%20show%20a)]. Parametri za kratki i dugi period su metaparametri u modelu. Za trening period su uzete prve 2 godine.

Sutrašnja cena se predviđa pomoću KNN-a. Uzimaju se današnji podaci (bez atributa 'volume') i traži se n najbližih dana u prethodnim podacima. Umesto običnih cena, koriste se returns-i (današnji returns je razlika između današnje i jučerašnje cene) [[5](https://www.investopedia.com/terms/r/return.asp#:~:text=A%20return%2C%20also%20known%20as%20a%20financial%20return%2C,derived%20from%20the%20ratio%20of%20profit%20to%20investment.)]. Koristi se euklidksa distanca. Od tih n dana uzimaju se 'close' returns-i njihovih narednih dana i pravi se težinska aritmetička sredina od njih gde su tegovi euklidkse distance od trenutne tačke. Ovim smo dobili sutrašnji return i na njega treba da dodamo današnju cenu kako bismo dobili predikciju sutrašnje cene.

Nakon što predvidimo sutrašnju cenu, tu predikciju trenutno dodajemo u poslednju linearnu regresiju u tom trenutku i proveravamo da li je greška tog modela veća od dosadašnje prosečne greške. Greška koja se posmatra je mape (mean absolute precentage error). [[6](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error#:~:text=Mean%20absolute%20percentage%20error%20is%20commonly%20used%20as,very%20intuitive%20interpretation%20in%20terms%20of%20relative%20error.)]

- Ukoliko jeste, postavljamo signal za trgovinu (signal će imati jačinu koeficijenta linearne regresije sačinjene od trenutnog dana i predviđanja sutrašnjeg dana), trgujemo (u sledećem pasusu je opisano trgovanje) i pravimo novi model od današnjeg i sutrašnjeg dana. U ovom momentu smemo da pogledamo sutrašnji dan jer smo već odlučili da li ćemo sutra trgovati ili ne (odluka da li se trguje ili ne se donosi dan pre nego što se zaista trguje i ne smeju da se koriste informacije o samom danu)
- Ukoliko nije, proveravamo da li je koeficijent trenutnog linearnog modela drugačijeg znaka od poslednjeg signala i da li je prošlo dovoljno dana od poslednjeg signala (adjusment period - metaparametar u modelu). Ukoliko jeste, pravimo novi signal sa trenutnim koeficijentom i trgujemo. U svakom slučaju na kraju, u poslednju linearnu regresiju dodajemo sutrašnju tačku. U ovom momentu iteriramo u sledeći dan.

Kada trgujemo, broj akcija se određuje na osnovu koeficijenta signala. Ukoliko je koeficijent negativan, prodajemo akcije a u koliko je pozitivan, kupujemo ih. Broj akcija je ograničen trenutnim kapitalom kada kupujemo, a zbirom trenutnog kapitala i vrednosti akcija kada prodajemo.

Mera kvaliteta modela koju koristimo je Sharpe ratio [[7](https://www.investopedia.com/terms/s/sharperatio.asp)]. To je prosek return-a kroz standardnu devijaciju return-a za neki vremenski period. Model takođe poredimo sa buy and hold strategijom [[8](https://www.investopedia.com/terms/b/buyandhold.asp)]. To je strategija gde na samom početku trgovine kupimo onoliko akcija koliko možemo i nikad ih ne prodamo.

Skup je podeljen na trening, validaciju i test (out of sample). Na validaciji se nađu najbolji metaparametri u odnosu na Sharpe ratio i ti parametri se iskoriste za trgovanje na test skupu. U modelu se ne trenira ponovo nad treningom i validaciji kada testiramo test skup već se samo nastavlja sa trgovinom gde se stalo. Iako prvi način ima više smisla u pogledu mašinskog učenja, za ovaj konkretan domen u praksi se češće radi na način iskorišćen u ovom modelu.

Napomena: U kodu se prvo izvršava predikcija svih tačaka u testu, pa pravljenje linearnih regresija na osnovu svih tačaka u testu, pa onda trgovanje po danima. Ovako je urađeno zbog logičnije podele koda i u redu je jer iako izračunamo predikcije i linearne regresije unapred, kada trgujemo ne gledamo u budućnost. Odnosno, pravimo se da te vrednosti još uvek nisu realizovane.

## Autor

Petar Zečević 1046/2020
