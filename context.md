quello che ho pensato é: gli agenti sanno, degli altri, la loro velocitá. di sé stessi anche l'energia. se sono in un gruppo, possono attaccare. attaccano valutando valore di payoff atteso: se attaccano, payoff sarebbe dato da un beneficio p_sustain*\sum_k p_follow(k)*U_mean(k+1), con p_sustain proporzionale a sigmoid((v_s-v_g)*k_a+bias). v_s sarebbe la velocitá che lágente puó sostenere contando distanza dalla fine e stanchezza, per arrivare alla fine prima. Il costo dovrebbe essere giá considerato, perché valuta se riuscirebbe ad arrivare alla fine. Se non attacca, il beneficio é dato da \sum_k U_mean(M), con M che sono i giocatori nel gruppo (Ha senso? sarebbe l'utility attesa se resta nel gruppo, quindi il posizionamento nella gara se rimane lí). Se qualcuno attacca, gli altri possono seguire. Seguono con probabilitá, simile a prima, data da p_sustain*\sum_k p_follow(k)*U_mean(k+1), con p_sustain che questa volta considera il beneficio del draft nel suo calcolo (quindi, seguire dovrebbe essere piú facile di attaccare) + bonus se cé teammate (?). In un gruppo, se non si attacca, si gioca public good game. ogni giocatore ha uno sforzo, numero da 0 a 1. La velocitá del gruppo dovrebbe essere una media pesata delle velocitá che possono sostenere gli agenti nel gruppo, pesata secondo lo sforzo impiegato. Peró, dovrebbero poter aumentare o diminuire lo sforzo in base alla stanchezza. Volendo, si puó fare compe implementato in precedenza, con sigmoid che dipende da distanza, stanchezza, presenza di amici. Gli agenti vengono considerati con benefici di draft se fanno parte di un gruppo. Se no, si considera come se chi si é staccato sta cercando di superare o simili.
Jun 21

Buona sintesi complessiva, molto più coerente delle versioni precedenti. Vediamo i punti chiave da chiarire/correggere prima che diventi codice.
Su p_sustain — la formula ha un problema di interpretazione

p_sustain ∝ sigmoid((v_s - v_g) · k_a + bias)

Qui v_s è la velocità che l'agente può sostenere fino alla fine, v_g presumo sia la velocità del gruppo (o forse la velocità richiesta per l'attacco?). Chiarisci: se v_s > v_g, l'agente può sostenere una velocità più alta di quella attuale del gruppo — questo dovrebbe corrispondere ad alta probabilità di successo dell'attacco (può effettivamente staccarsi e arrivare). La formula con (v_s - v_g) positivo che alimenta una sigmoid dà effettivamente p_sustain alto in quel caso — quindi il segno è coerente, bene.

Però attenzione: dici "v_s sarebbe la velocità che l'agente può sostenere contando distanza dalla fine e stanchezza, per arrivare alla fine prima" — questo significa che v_s è già un'ottimizzazione interna (la velocità ottimale per minimizzare il tempo di arrivo dato quanta energia resta). Questo è un calcolo non banale (è un problema di pacing, letteratura ricca su questo, es. Olds che hai già citato) — assicurati di avere una formula chiusa o un'approssimazione semplice per v_s, altrimenti rischi di dover risolvere un problema di ottimizzazione vincolata ad ogni timestep per ogni agente, che è costoso computazionalmente.
Sul confronto "non attacca" — qualcosa non torna nel confronto dimensionale

beneficio_attacco = p_sustain · Σ_k p_follow(k) · U_mean(k+1)
beneficio_non_attacca = Σ_k U_mean(M)

Qui c'è un'asimmetria che sembra un errore di formulazione: nel caso "attacca" hai una sommatoria su k (numero di possibili follower, con la sua distribuzione di probabilità) — ma nel caso "non attacca" hai scritto Σ_k U_mean(M), con M fisso (dimensione attuale del gruppo) e una sommatoria su k che non ha senso se M non dipende da k. Presumo sia un errore di notazione e intendessi semplicemente:

beneficio_non_attacca = U_mean(M)

cioè il payoff atteso di rimanere nel gruppo di dimensione attuale M, senza sommatoria — perché se non attacchi, non c'è incertezza su quanti "ti seguono" (resti dove sei). Confermalo.
Sul "seguire" — buona intuizione sul draft, ma il bonus team va specificato meglio

Hai messo + bonus se c'è teammate (?) con un punto interrogativo giusto — è il punto debole. Ti ricollego a quanto discusso prima: il bonus per un compagno che attacca non dovrebbe essere semplicemente additivo fisso, ma dovrebbe modulare l'intera valutazione, perché cambia la natura del payoff: se seguo un compagno, anche se lui vince e non io, ottengo comunque il bonus di team. Quindi più correttamente:

beneficio_follow = p_sustain_follow · Σ_k p_follow(k) · [
    U_mean_personale(k+1) + [chi_attacca_è_teammate] · p_win_attaccante(k) · bonus_team
]

Questo è diverso da un bonus additivo semplice — qui il bonus dipende anche da quanto è probabile che l'attaccante (il tuo compagno) vinca date le condizioni, non solo dal fatto binario che sia un compagno.
Sul public goods game dentro al gruppo — un punto tecnico importante manca

Hai detto: "la velocità del gruppo dovrebbe essere una media pesata delle velocità sostenibili, pesata secondo lo sforzo" — coerente con quanto avevamo già discusso. Ma noti tu stesso "dovrebbero poter aumentare o diminuire lo sforzo in base alla stanchezza" — qui serve specificare la direzione causale: lo sforzo (input strategico) determina quanto consumi energia, ma è anche vero che quanta energia hai già consumato (stanchezza) dovrebbe influenzare quanto sforzo sei disposto a metterci ora. Questo crea una dipendenza naturale e legittima (non circolare, perché la stanchezza al tempo t è dovuta a sforzi passati, non a quello corrente) — basta che nel codice usi la stanchezza prima dell'aggiornamento per decidere lo sforzo di questo timestep, esattamente come il pattern step/advance che avevi già.
Punto critico non ancora risolto: cosa succede quando un agente attacca con successo (nessuno lo segue) o fallisce (scoppia)

Hai descritto le decisioni (attaccare, seguire, sforzo nel gruppo) ma non il meccanismo fisico di cosa succede dopo che un agente decide di attaccare. Specificamente:

    Se attacca e nessuno segue: si stacca dal gruppo, va a v_s (velocità sostenibile individuale) — ma per quanti timestep? Fino alla fine della gara, o ricalcola v_s ogni timestep in base a quanto gli resta?
    Se attacca e viene seguito da k persone: si forma un nuovo gruppo (lui + i k follower), che gioca lo stesso public goods game del gruppo principale, solo più piccolo? Presumo sì, per consistenza, ma confermalo.
    Cosa succede al "gruppo principale" rimasto (N - k - 1 persone)? Continuano il loro public goods game come se niente fosse, ignorando l'attacco? O dovrebbero anche loro rivalutare se inseguire il gruppo dell'attacco (un secondo livello di decisione: il gruppo principale, nel suo complesso, decide se aumentare il ritmo per richiudere il gap)?

Questo ultimo punto è importante perché altrimenti il gruppo principale non reagisce mai agli attacchi (a meno che i singoli decidano individualmente di "seguire", ma seguire fisicamente richiede staccarsi dal gruppo principale e raggiungere quello davanti, che è un'azione diversa da "il gruppo nel suo complesso accelera per non perdere terreno").
Sul "chi si è staccato cerca di superare o simili" — questo è vago, va specificato

Hai chiuso con "se no, si considera come se chi si è staccato sta cercando di superare o simili" — questo sembra riferirsi al caso di un agente isolato che, vedendo un gruppo davanti, decide se inseguirlo. Ma questo è concettualmente uguale al "seguire" (stesso meccanismo, target diverso: invece di seguire un attacco appena partito, insegui un gruppo già consolidato più avanti). Se è così, puoi riusare la stessa formula di utility per "seguire/raggiungere" sia nel caso di un attacco fresco sia nel caso di un gruppo già lontano — semplifica il modello, non serve un quinto meccanismo separato.
Riepilogo dei punti da chiudere prima di formalizzare in pseudocodice

    Confermare che beneficio_non_attacca = U_mean(M) senza sommatoria (probabile typo)
    Specificare una forma chiusa o approssimazione per v_s (velocità sostenibile ottima dato distanza e stanchezza) — questo è un sotto-problema di pacing che richiede la sua formula
    Formalizzare il bonus team come modulazione probabilistica, non additiva fissa
    Decidere se il "gruppo principale" reagisce collettivamente agli attacchi o solo tramite decisioni individuali di chi insegue
    Confermare che "inseguire un gruppo lontano" riusa la stessa formula di "seguire un attacco fresco"

Vuoi che affrontiamo prima il punto 2 (la formula di v_s), dato che è probabilmente il pezzo matematicamente più delicato e da cui dipendono gli altri calcoli?

    si era un typo. 2. per ora sarebbe W_prime=P_tot=v(C_r+kv^2), che non é chiusa. 3. cioé? 4. se il gruppo reagisce agli attacchi come un unica entitá, viola o no i requisiti del progetto? 5. confermo. DOMANDA: formulando cosí il modello, soddisferebbe tutti i prerequisits?

12:01 AM

Affronto tutto in ordine, poi do la valutazione complessiva sui requisiti.
2. La formula di v_s — chiariamo cosa significa "non chiusa" e come risolverla

W_prime = P_tot = v(Cr + k·v²)

Questa è l'equazione di potenza-velocità in stato stazionario, ma per trovare v_s (la velocità sostenibile dato quanta energia/tempo resta), devi risolvere un problema diverso: non "che potenza serve per questa velocità" ma "qual è la velocità che, mantenuta costante, mi fa esaurire esattamente l'energia residua esattamente alla distanza che manca".

Questo è un problema di ottimizzazione vincolata: dato W_prime_residuo e distanza_residua, trova v tale che il tempo per percorrere quella distanza a velocità v esaurisca esattamente l'energia.

tempo_residuo = distanza_residua / v
energia_consumata = (P(v) - CP) · tempo_residuo     [se P(v) > CP]
vincolo: energia_consumata = W_prime_residuo

→ (v·(Cr + k·v²) - CP) · (distanza_residua / v) = W_prime_residuo

Questa è un'equazione cubica in v (espandendo, hai termini in v² e v³ moltiplicati per 1/v, quindi resta un polinomio di terzo grado) — risolvibile in forma chiusa con la formula di Cardano (esattamente come hai già fatto per aerobic_speed nel codice precedente, che era una cubica simile). Non è "non chiusa" nel senso di impossibile da risolvere analiticamente — è chiusa, ma richiede la stessa tecnica (Cardano) che hai già usato altrove nel tuo modello. Buona notizia: hai già il codice per questo tipo di equazione, basta adattare i coefficienti.
3. Il bonus team come modulazione — un esempio concreto

Intendevo questo: invece di
python

U_follow = qualcosa + bonus_team_fisso   # additivo, sempre la stessa quantità

vuoi
python

U_follow = qualcosa + p_win_attaccante_stimata · bonus_team   # il bonus è scalato dalla probabilità che il compagno vinca davvero

La differenza pratica: con il bonus fisso, seguire un compagno è sempre "ugualmente attraente" indipendentemente da quanto sia probabile che vinca. Con il bonus modulato, seguire un compagno che ha alta probabilità di vincere (perché è forte, o perché la situazione è favorevole) è più attraente che seguire un compagno improbabile vincitore — il che è più realistico: non ti sacrifichi per un compagno che ha 2% di possibilità di vincere tanto quanto per uno che ha 40%.
4. Il gruppo che reagisce come unica entità — qui c'è un problema reale con i requisiti

Sì, violerebbe il punto 9 (no controllo centrale) se implementato come "il gruppo, in quanto entità, decide collettivamente di accelerare". Un gruppo non può essere un agente decisionale nel tuo modello — è una categoria emergente che tu misuri, non un'entità con una propria funzione di decisione.

La soluzione corretta: la reazione collettiva deve emergere dalla somma di decisioni individuali. Ogni membro del gruppo principale, vedendo l'attacco, decide individualmente se aumentare il proprio sforzo di cooperazione (il parametro c_i del public goods game) — se molti decidono indipendentemente di aumentare c_i in risposta all'attacco osservato, il gruppo nel suo complesso accelera come conseguenza aggregata, non come decisione presa "dal gruppo".

Questo è in realtà elegante e già coerente con quello che hai: la decisione "quanto sforzo metto nel public goods game" può già dipendere, tra i suoi input, da "c'è un attacco in corso da parte di chi non è nel mio gruppo" — è solo un input aggiuntivo alla stessa sigmoid/calcolo di utility che già usi per c_i, non un meccanismo nuovo.
5. Confermato, bene — risparmi un meccanismo separato.
DOMANDA PRINCIPALE: questo modello soddisfa tutti i requisiti?

Andiamo punto per punto, con onestà sui rischi residui.

1. Entità discrete con identità ✓ — Ogni ciclista è un agente Mesa distinto, con team_id, parametri fisici propri. Soddisfatto chiaramente.

2. Stati interni che cambiano nel tempo ✓ — Energia (W'), posizione, velocità, credenze bayesiane (su p_follow/threshold). Soddisfatto.

3. Spazialmente localizzati in spazio/rete/topologia esplicita ✓ — Hai la retta 1D con posizione x_i. Soddisfatto, anche se semplice.

4. Percepiscono e interagiscono con l'ambiente ✓ — Osservano velocità altrui, posizioni, formano gruppi per prossimità. Soddisfatto, a patto che l'osservazione resti locale (solo vicini nel raggio percettivo, non stato globale dell'intera gara) — verifica questo nell'implementazione.

5. Bounded rationality, non ottimizzazione perfetta ✓ — Hai stime rumorose (p_follow stimata, non nota), informazione incompleta sulla fatica altrui, apprendimento bayesiano invece di calcolo esatto. Ben soddisfatto, è uno dei punti più forti del tuo design attuale.

6. Bias cognitivi: risk aversion e loss aversion ⚠️ — Qui hai ancora un buco. Avevamo discusso λ_risk (penalità su varianza) e una funzione di utility asimmetrica per loss aversion (prospect theory), ma nelle ultime formulazioni che hai scritto non vedo questi termini esplicitamente presenti. Le formule attuali di utility sono valori attesi lineari (Σ p(k)·U(k)), senza penalità di varianza né asimmetria guadagni/perdite. Devi reintrodurli esplicitamente prima di considerare questo requisito soddisfatto — non è implicito nella struttura attuale.

7. Possono imparare/adattarsi nel tempo ✓ — Hai sia l'apprendimento bayesiano dentro la gara (su p_follow/threshold) sia il replicator dynamics tra gare (sui parametri di disposizione). Doppio livello, ben soddisfatto.

8. Interazioni strategiche con struttura game-theoretic formale ✓ — Public goods game per la cooperazione (con free-riding esplicito, confermato anche dalla letteratura che hai trovato), gioco di attacco/inseguimento con payoff dipendenti dalla risposta altrui (binomiale su k follower). Ben strutturato e formalizzabile.

9. No controllo centrale ⚠️ — Dipende dall'implementazione, attenzione al punto 4 appena discusso. Se rispetti la correzione che abbiamo appena fatto (il gruppo non decide come entità, solo individui decidono), allora è soddisfatto. Ma è un punto fragile — verifica ogni meccanismo per assicurarti che nessuna decisione sia presa "a livello di gruppo" invece che aggregata da scelte individuali.

10. Comportamento emergente non triviale ✓ — Hai due target espliciti e ben scelti (differenziazione leader/domestique da parametri fisici, non da regole; timing degli attacchi che emerge dall'apprendimento, non imposto). Plausibile, ma lo dimostri solo a posteriori, eseguendo il modello — non è qualcosa che puoi dichiarare soddisfatto solo dal design, va verificato empiricamente sui risultati.

11. Sensitivity analysis robusta — Non è una proprietà del modello stesso, ma del tuo piano di analisi successivo. Hai già identificato i 5 parametri principali; soddisfacibile a valle, non è un problema di design ora.
Verdetto

Il modello soddisfa bene 8 punti su 11 chiaramente, con un buco esplicito (punto 6, bias cognitivi mancanti nelle ultime formule) e due punti che dipendono dall'esecuzione/implementazione corretta più che dal design (punto 9, controllo centrale — rischio gestibile seguendo la correzione discussa; punto 10, emergenza — verificabile solo a posteriori).

Priorità immediata: reintrodurre esplicitamente risk aversion e loss aversion nelle formule di utility, altrimenti un valutatore esterno del progetto segnalerebbe questo come requisito non soddisfatto in modo diretto e verificabile nel codice/formule.

Vuoi che formalizziamo insieme dove esattamente inserire questi due bias nelle formule di utility che hai già scritto per attacco/seguire/cooperare?
Files hidden in shared chats

quindi la cooperazione potrebbe essere data dallóbiettivo che ho, ovvero in questo caso raggiungere il gruppo che vola. quindi, calcolo, sapendo quanto sforzo é stato fatto in precedenza, quanto ne devo fare per ottenere il mio obiettivo. poi é una sorta di el farol problem, in cui gli agenti cercano di mettere contributi per far raggiugnere il proprio obiettivo e risparmiare energia
12:42 AM

L'intuizione di El Farol è interessante e in parte calza, ma c'è una distinzione importante da fare prima di adottarla, perché El Farol ha una struttura specifica che non corrisponde del tutto a quello che stai descrivendo.
Cos'è El Farol problem esattamente

Il problema classico: N persone decidono indipendentemente se andare al bar; se troppe vanno (sopra una soglia), l'esperienza è negativa per tutti (troppo affollato) e converrebbe non essere andati; se poche vanno, è positivo per chi è andato. È un gioco di coordinamento con disincentivo alla congestione, dove il payoff di un'azione diminuisce con il numero di altri che fanno la stessa azione.
Dove il tuo problema è simile a El Farol

C'è una vera analogia nel gioco dell'inseguimento/attacco: se troppi seguono l'attaccante, il vantaggio individuale di essere nel breakaway si dilude (lo avevi già capito tu stesso con U_mean(k+1) che decresce in k) — più gente segue, meno vantaggioso è per ciascuno. Questo è strutturalmente analogo a El Farol: il payoff della tua azione (seguire/attaccare) dipende negativamente da quanti altri fanno la stessa scelta.
Dove invece NON è El Farol — il public goods game della cooperazione

Per la cooperazione dentro il gruppo (chi tira, sforzo nel raggiungere un obiettivo comune), la struttura è diversa: qui il beneficio aumenta con quanti cooperano (più gente tira, più il gruppo va veloce, più vantaggio per tutti), non diminuisce. Questo è l'opposto della logica di congestione di El Farol — è un classico public goods game, che è quello che stavi già usando correttamente.

Quindi: se stai proponendo El Farol per il calcolo dello sforzo di cooperazione (quanto contribuire per raggiungere il gruppo che vola), probabilmente non è la cornice giusta — il problema lì non è "troppa gente che fa la stessa cosa rovina il risultato per tutti", è "serve un contributo aggregato sufficiente, e ognuno preferirebbe che lo dessero gli altri" (free-rider problem classico, non El Farol).
Riformulazione di quello che hai descritto, senza El Farol

Quello che dici — "calcolo, sapendo quanto sforzo è stato fatto in precedenza, quanto ne devo fare per ottenere il mio obiettivo, risparmiando energia" — è in realtà un problema di contribuzione marginale a un obiettivo con soglia, che è leggermente diverso sia da El Farol sia dal public goods game puro. È più vicino a un problema tipo:

obiettivo: raggiungere il gruppo che vola, che richiede un certo "lavoro totale" W_richiesto 
           (es. funzione del gap di velocità/distanza da chiudere)

ogni agente i, vedendo quanto lavoro è già stato fatto dagli altri (osservabile: la velocità 
attuale del gruppo che insegue, risultato aggregato dei contributi)
decide il proprio contributo c_i per il prossimo istante, bilanciando:
   - quanto si avvicina all'obiettivo (beneficio)
   - quanta energia spende (costo, pesato da λ_cost)

Questo è effettivamente un public goods game con soglia/obiettivo esplicito (a volte chiamato "threshold public goods game" in letteratura, distinto dal public goods game lineare classico) — è una variante nota e ben studiata, diversa da El Farol ma concettualmente vicina a quello che hai in mente.
La parte di El Farol che potresti voler recuperare comunque

C'è un aspetto di El Farol che potrebbe interessarti per un motivo diverso: in El Farol, gli agenti usano previsioni eterogenee basate su regole induttive semplici (ognuno ha una sua euristica su "quanta gente andrà") per decidere — è un classico esempio di bounded rationality con eterogeneità di modelli mentali. Questo si collega bene al tuo requisito 5/6 (bounded rationality, no informazione perfetta): ogni agente, non sapendo esattamente quanto stanno contribuendo gli altri (energia è privata), usa una stima/euristica propria basata solo su segnali osservabili (velocità del gruppo osservata) per decidere il proprio contributo. Questa parte di "decisione euristica sotto incertezza con previsione eterogenea" è il contributo di El Farol che vale la pena citare, anche se la struttura di payoff sottostante resta quella del threshold public goods game, non quella di congestione.
Proposta concreta per formalizzare il tuo contributo c_i con questa logica
python

def decide_contribution(self, observed_group_speed, distance_to_target_group, my_fatigue):
    # stima quanto lavoro è già stato fatto dagli altri, 
    # osservando solo l'esito aggregato (velocità corrente), non i contributi individuali altrui
    work_done_estimate = self.infer_aggregate_effort(observed_group_speed)
    
    work_remaining = self.compute_required_work(distance_to_target_group)
    
    my_needed_contribution = max(0, work_remaining - work_done_estimate) / self.n_group_members_estimate
    
    # bilancia con il costo personale, scalato da quanto sono già stanco
    c_i = self.balance_contribution_vs_fatigue(my_needed_contribution, my_fatigue)
    
    return c_i

Questo cattura quello che descrivevi: l'agente stima cosa serve per raggiungere l'obiettivo (raggiungere il gruppo che vola), guarda quanto è già stato fatto (inferendolo da segnali osservabili, non conoscenza diretta degli altri), e decide il proprio contributo bilanciando con la propria stanchezza — senza violare l'informazione privata sull'energia altrui.

Ti sembra che questa via (threshold public goods game + stima euristica bounded-rational, citando El Farol solo per l'aspetto di previsione eterogenea) catturi meglio quello che avevi in mente rispetto a un El Farol problem puro?

allínizio della gara, un agente vede un altro e decide di seguirlo. come dovrebbe decidere? cosa valuta? essendo línizio della gara, é un pó difficile parlare di proabi=bilitá di vittoria
12:51 AM

Buona osservazione — all'inizio gara, "probabilità di vincere" è un concetto quasi vuoto (troppe incognite, troppo lontano nel tempo per essere informativo). Serve un criterio diverso, più locale e a breve termine.
Il problema con usare p_win all'inizio

Le formule che abbiamo costruito finora (binomiale su follower, U_mean(k+1), ecc.) hanno senso quando l'esito è relativamente prossimo e quindi stimabile — vicino alla fine, "vincere" è un evento concreto e vicino. All'inizio gara, chiedere a un agente "quanto vale seguire questo tizio in termini di probabilità di vittoria finale" richiede propagare l'incertezza su centinaia di km e ore di gara — rumore così alto che la stima non porta informazione utile.
Cosa valuta davvero un ciclista all'inizio gara (intuizione realistica)

Nella realtà, all'inizio di una gara i ciclisti non valutano "vincerò se seguo questo?" — valutano qualcosa di molto più locale e immediato: "vale la pena spendere energia ora per questo movimento specifico?" Le determinanti tipiche sono:

    Chi è quella persona (è una minaccia da non lasciar scappare, è un compagno, è un nessuno che non conta)
    Quanto costa seguirlo ora (rispetto a quanta energia ho, e quanta gara resta — molto presto, ogni dispendio energetico ha un costo opportunità alto perché c'è ancora tantissima strada)
    Se altri lo seguono già (se molti altri si muovono, è un segnale che "vale" inseguire; se nessuno si muove, forse è un falso allarme)

Riformulazione: usa un orizzonte locale, non l'esito finale

Invece di stimare l'impatto su p_win_finale, usa una utility locale e incrementale, che valuta l'effetto immediato della scelta su un orizzonte breve (es. "dove sarò rispetto al gruppo nei prossimi minuti"), pesato dal costo energetico immediato:
python

def utility_follow_early_race(self, other_agent, observed_speed_other):
    # quanto costa seguire ORA, in termini di frazione di riserva energetica
    cost = self.energy_cost(target_speed=observed_speed_other) / self.W_prime_max
    
    # quanto "vale" non perdere il contatto - proxy locale, non probabilità di vittoria
    threat_level = self.estimate_threat(other_agent)  
    # es. funzione di: è compagno di squadra noto? quanti altri lo stanno seguendo? 
    #     quanto è "anomala" la sua accelerazione rispetto al ritmo gruppo?
    
    benefit = threat_level  # NON moltiplicato per p_win, è un proxy di rilevanza immediata
    
    return benefit - lambda_cost * cost

threat_level qui sostituisce il ruolo che prima aveva p_win, ma è una stima di rilevanza tattica immediata, non di vittoria finale — molto più appropriato per l'inizio gara.
Una via alternativa più elegante: usa lo stesso framework ma con un "obiettivo" più vicino

Ricollegandoci a quanto avevamo appena discusso (threshold public goods game, raggiungere un obiettivo), potresti trattare "seguire all'inizio gara" come lo stesso tipo di decisione che già hai per "raggiungere il gruppo che vola" — solo che l'obiettivo non è "vincere", è "non perdere posizione relativa rispetto agli altri". Questo è coerente con tutto il framework che hai già costruito, senza bisogno di un meccanismo nuovo apposta per l'inizio gara:

U_follow = beneficio_di_non_perdere_terreno - costo_energia

dove "non perdere terreno" può essere quantificato come una proxy semplice: la distanza che si aprirebbe rispetto a chi sta accelerando, se non lo seguo, normalizzata in qualche modo (es. come frazione della distanza totale di gara, così il costo psicologico/strategico di un gap che si apre presto è automaticamente scontato rispetto a uno vicino alla fine, perché c'è più tempo per richiuderlo).
Punto chiave: la differenza tra "minaccia vera" e "rumore"

All'inizio gara, gran parte delle accelerazioni sono rumore tattico (sondaggi, finte, attacchi destinati a fallire) — non ogni movimento merita una risposta. Qui torna utile l'apprendimento bayesiano che avevi già in mente: l'agente può mantenere una stima (aggiornata nel corso della gara, e nel tempo tramite replicator dynamics tra gare) di quanto è probabile che un'accelerazione iniziale sia "vera" (porterà a un vantaggio reale) vs rumore. Early in gara, il prior dovrebbe essere "probabilmente rumore" (molte accelerazioni fittizie), e solo accelerazioni anomale (velocità molto sopra il gruppo, sostenute per più timestep) aggiornano la credenza verso "questa potrebbe essere seria".

Ti sembra più sensato ancorare la decisione iniziale a "non perdere terreno relativo" (proxy locale, riusa il framework che hai) invece che forzare il calcolo di probabilità di vittoria fin dall'inizio gara?

potrei fare: se vedo qualcuno piú veloce di me, lo voglio raggiungere (quello piú vicino a me). calcolo se riesco a sostenere il suo passo. accelero (di quanto? questo puó essere un parametro) per raggiungerlo e formiamo un gruppo. Lui dovrebbe poter decidere se scappare o no?
12:59 AM

Buona semplificazione — riduce drasticamente la complessità e ha senso comportamentale. Vediamo i punti da definire con precisione prima che diventi codice.
La logica generale funziona, ma "quello più vicino a me" merita un chiarimento

Se intendi "il più vicino in posizione spaziale", va bene ed è semplice. Ma se più agenti vicini hanno velocità diverse, dovresti specificare: l'agente guarda tutti quelli nel suo raggio percettivo e sceglie il più vicino tra quelli più veloci di lui? O il più veloce in assoluto, indipendentemente dalla distanza? Sono criteri diversi — "il più vicino" privilegia facilità di raggiungimento, "il più veloce" privilegia il potenziale guadagno. Ti consiglio "il più vicino tra quelli più veloci" come hai detto, perché è più coerente con un agente bounded-rational che non sta facendo un'ottimizzazione globale su tutti i possibili target.
"Calcolo se riesco a sostenere il suo passo" — qui riusi la formula di v_s che avevamo già
python

puo_sostenere = (velocità_osservata_altro <= v_s_self)

dove v_s_self è la velocità sostenibile fino alla fine che avevamo discusso con l'equazione cubica. Se sì, ha senso provare ad agganciarsi. Se no, l'agente sa già che il tentativo è destinato a fallire (scoppierà prima) — quindi non ci proverebbe nemmeno, a meno che tu voglia introdurre un margine di errore/eccesso di confidenza (un bias cognitivo interessante, ma probabilmente non necessario ora).
"Accelero di quanto" — questo non dovrebbe essere un parametro libero, ecco perché

Se tratti "quanto accelero" come parametro libero aggiuntivo, aggiungi un grado di libertà che in realtà è già determinato dalla fisica: per raggiungere qualcuno che è a distanza d e va a velocità v_target, l'agente deve temporaneamente andare più veloce di v_target per chiudere il gap in un tempo finito. Più accelera, più velocemente lo raggiunge ma più energia consuma nello scatto.

Questo è in realtà un'altra decisione strategica (quanto aggressivo essere nello scatto), non un parametro fisso di design — coerente con tutto il framework, dovrebbe essere parte di un calcolo di utility: scatto forte (raggiungo rapidamente, consumo più energia nello scatto) vs scatto debole (impiego più tempo, durante il quale l'altro potrebbe accelerare ulteriormente o il gap potrebbe anche chiudersi naturalmente se lui rallenta).

Soluzione semplice se vuoi evitare un'altra decisione complessa: fissa una regola semplice non parametrica, tipo "accelera fino al massimo sostenibile per il tempo minimo necessario a chiudere il gap, poi torna a v_target" — qui non c'è un parametro libero, è dettato dalla fisica e dall'obiettivo (chiudere il gap il prima possibile dato che hai deciso di provarci). Questo è coerente con bounded rationality semplice: una volta deciso di inseguire, lo fai nel modo più diretto, senza un ulteriore livello di ottimizzazione su "quanto forte accelerare".
"Formiamo un gruppo" — attenzione, qui c'è di nuovo il rischio di azione congiunta/centralizzata

Hai scritto "formiamo un gruppo" come se fosse un'azione bilaterale concordata — ma ricorda il punto 9. La formazione del gruppo deve essere una conseguenza emergente, non un'azione congiunta decisa da entrambi insieme. Il modo corretto:

Agente A decide, individualmente, di accelerare per raggiungere B (la sua decisione, presa solo guardando B)
Agente B, indipendentemente, decide cosa fare quando nota A che si avvicina (scappare o tollerare)
Se A raggiunge B (distanza < 3m) E B non è scappato efficacemente, allora per definizione (la tua soglia) sono nello stesso gruppo

Il "gruppo" emerge dalla misurazione di prossimità, non da una decisione condivisa — esattamente come discusso prima. Nessuno dei due "decide di formare un gruppo": A decide di inseguire, B decide come reagire, e il gruppo è semplicemente l'etichetta che applichi tu in base al risultato di queste due decisioni indipendenti.
"Lui dovrebbe poter decidere se scappare o no" — sì, ed è importante per evitare un altro problema

Hai ragione, e questo è cruciale: se B non ha alcuna possibilità di reagire (è passivo, viene semplicemente raggiunto), allora l'interazione non è più strategica nel senso del punto 8 — è solo fisica unilaterale. Dando a B la scelta di accelerare per scappare (se può sostenerlo) o di accettare la fusione (se preferisce, magari perché gli conviene il gruppo più grande), recuperi la struttura di gioco a due (o più) giocatori con scelte simultanee/sequenziali, che è quello che serve.

La decisione di B userebbe lo stesso tipo di calcolo che hai già: B valuta U_scappare (continua/aumenta la sua velocità, sostenibile se ha abbastanza margine) vs U_accetta_fusione (rallenta o mantiene il passo, si forma il gruppo, quali sono i benefici/costi di questo nuovo gruppo per lui).
Schema riassuntivo della sequenza

1. A osserva B (più vicino tra i più veloci), valuta se può sostenere v_B
2. Se sì: A decide se vale la pena (utility positiva netta) provare ad agganciarsi
3. A accelera al massimo sostenibile per il tempo minimo necessario a chiudere il gap
4. NEL FRATTEMPO, B osserva A che si avvicina, decide: scappare (se conviene e può) o tollerare
5. Esito determinato dalla combinazione: se B scappa con successo, gap si riapre, A può rivalutare 
   (riprovare, desistere); se B tollera o non riesce a scappare, distanza scende sotto soglia, 
   si misura il "gruppo" risultante

Questo è pulito, decentralizzato, e strategico. Vuoi che formalizziamo le utility di A (step 2) e di B (step 4) con le formule esplicite, magari riusando gli elementi già costruiti (λ_cost, stima locale di rilevanza, ecc.)?

non saprei come formalizzare le utility di b per decidere se scappare o tollerare
1:04 AM

Pensiamoci da zero, partendo da cosa B osserva e cosa gli importa davvero in quel momento.
Cosa sa B nell'istante della decisione

B vede A che si avvicina (osserva la sua velocità, distanza, magari il team_id se visibile — es. divisa della squadra). B non conosce l'energia interna di A, ma può inferire qualcosa dal fatto che A si stia muovendo a una certa velocità sostenuta.

B conosce sé stesso: la propria energia residua, la propria v_s (velocità sostenibile fino alla fine), la distanza dalla fine.
Le due opzioni di B, in termini di conseguenze concrete

Tollerare (lasciarsi raggiungere): B continua al suo ritmo attuale. Conseguenza: si forma un gruppo con A (e chiunque altro converga). B guadagna il beneficio di draft/cooperazione del gruppo più grande, ma diluisce la sua posizione relativa (non è più "avanti da solo", è uno tra N).

Scappare (accelerare per mantenere il vantaggio): B aumenta velocità per restare avanti rispetto ad A. Conseguenza: se ci riesce, mantiene/aumenta il vantaggio posizionale, ma paga un costo energetico extra (sta accelerando sopra il suo ritmo "naturale" di quel momento).
La formalizzazione più semplice: confronta i due scenari con la stessa logica che hai già

Hai già tutti i pezzi — è lo stesso schema usato per "attaccare vs restare nel gruppo":
python

def utility_tolerate(self):
    # se tollero, mi fondo con A (e possibili altri), divento parte di un gruppo di dimensione stimata
    M_estimated = self.estimate_resulting_group_size()  # 2, o più se ci sono altri vicini convergenti
    return U_mean(M_estimated)  # stesso U_mean(M) che usavi per "non attacca, resta nel gruppo"

def utility_escape(self):
    # se scappo, devo sostenere una velocità più alta di quella di A per non farmi raggiungere
    p_sustain_escape = sigmoid((self.v_s - v_A_observed) * k_a + bias)
    
    # se scappo con successo, resto solo (o avanti) - stesso tipo di payoff di un "attacco" riuscito
    benefit_if_success = U_mean(1)  # da solo, il massimo payoff posizionale possibile in quel momento
    
    cost = self.energy_cost(target_speed=v_A_observed + margin) / self.W_prime_max
    
    return p_sustain_escape * benefit_if_success - lambda_cost * cost

# B decide:
if utility_escape() > utility_tolerate():
    scappa
else:
    tollera

Questo è esattamente la stessa struttura logica che hai già costruito per la decisione di attaccare — non è un meccanismo nuovo, è un riuso: "scappare da chi si avvicina" è concettualmente identico ad "attaccare per allontanarsi", solo innescato da uno stimolo diverso (qualcuno che si avvicina, invece di una decisione spontanea).
Perché questo riuso ha senso e ti semplifica la vita

Hai già p_sustain (sigmoid sulla differenza di velocità sostenibile), hai già U_mean(M) (payoff atteso per dimensione di gruppo), hai già lambda_cost. Non serve nessuna formula nuova — basta riconoscere che "B decide se scappare da A" è la stessa famiglia di decisione di "un agente isolato decide se attaccare", con il target/trigger diverso (qui è reattivo, innescato dall'osservare A avvicinarsi; prima era proattivo/spontaneo).
L'unica cosa nuova: la stima della dimensione del gruppo risultante se tollera

estimate_resulting_group_size() è il pezzo che devi specificare — una stima semplice potrebbe essere: 1 (B) + 1 (A) + numero di altri agenti osservati nelle vicinanze che si stanno muovendo nella stessa direzione/velocità convergente. Anche questa è informazione osservabile localmente (posizioni e velocità di chi è nel raggio percettivo), non richiede sapere lo stato interno di nessuno.

Ti sembra che questo riuso della struttura "attacco" per il caso "scappare" risolva il problema, o preferisci che differenziamo i due casi con qualche elemento aggiuntivo specifico (es. B ha meno tempo per decidere rispetto a chi attacca spontaneamente, quindi forse meno accuratezza nella stima)?

potrei fare che quando qualcuno si stacca lo fa per valutare se raggiungere qualcuno davanti a sé o, se non cé nessuno davanti, il traguardo.
1:09 AM

Sì, ha senso ed è una bella unificazione — riduce ulteriormente il numero di meccanismi distinti nel modello. Vediamo l'implicazione precisa.
L'idea: un solo "target" generico, non distinzione tra casi

Invece di avere "attaccare" (nessun bersaglio specifico, solo provare ad andare forte) e "raggiungere qualcuno" come due meccanismi separati, unifichi tutto sotto: ogni agente, quando valuta di staccarsi, ha sempre un target — che è il prossimo gruppo/agente davanti a sé, oppure, se è già il primo, il traguardo stesso.
python

def get_target(self):
    next_group = self.model.get_group_ahead(self)
    if next_group is not None:
        return next_group  # target: raggiungere chi è avanti
    else:
        return "finish_line"  # target: il traguardo stesso

Perché questo è elegante

Risolve un buco che avevi nel modello: prima, "attaccare" per chi era già il leader della corsa non aveva un target chiaro (cosa significa attaccare se sei già davanti a tutti?) — con questa formulazione, il leader della gara ha semplicemente come target il traguardo, e la sua decisione di "attaccare" diventa "decido di aumentare il ritmo verso il traguardo", usando esattamente la stessa logica di utility (p_sustain, costo energetico) che usavi per raggiungere un altro gruppo, solo con target = fine gara invece che target = altro agente.
Cosa cambia nella formula di utility

Per il caso "target = altro gruppo/agente", hai già tutto: p_sustain rispetto alla velocità del target, beneficio in base a quanti ti seguirebbero, ecc.

Per il caso "target = traguardo", la formula si semplifica perché non c'è incertezza su "quanto è lontano/veloce il target" (la posizione del traguardo è nota con certezza, è la distanza rimanente) — diventa puramente una questione di pacing: dato che resta una distanza D, qual è la strategia di velocità ottimale per arrivare il prima possibile senza esaurire l'energia prima del traguardo? Questo è esattamente il problema che avevamo risolto con l'equazione cubica per v_s.
python

def utility_attack_toward_finish(self):
    v_optimal = self.compute_v_s(distance=self.distance_to_finish)
    benefit = U_mean(1)  # se nessuno mi raggiunge nel frattempo, vinco da solo
    
    # qui la "minaccia" non è un trigger esterno ma il rischio che altri dietro mi raggiungano
    p_caught = self.estimate_probability_being_caught()  
    
    cost = self.energy_cost(v_optimal) / self.W_prime_max
    
    return (1 - p_caught) * benefit - lambda_cost * cost

Nota che qui serve comunque un termine di rischio — non "essere seguito" (non ha senso se sei già da solo davanti), ma "essere raggiunto da dietro" — che è concettualmente l'inverso: non stimi p_follow di chi ti segue volontariamente, stimi la probabilità che il gruppo dietro (o un singolo) ti recuperi prima del traguardo (qui torna utile la Chapatte's Law come riferimento empirico per calibrare/validare questa stima).
Conseguenza pratica: elimini un meccanismo, ma aggiungi un sotto-caso da formalizzare

Hai ridotto "attaccare" e "inseguire un gruppo" a un solo meccanismo generico ("muoviti verso il tuo target, che sia un gruppo o il traguardo") — ottima semplificazione concettuale. Ma il prezzo è che devi formalizzare bene p_caught (probabilità di essere ripreso da dietro quando il target è il traguardo), che prima non avevi bisogno di esplicitare separatamente.

p_caught dovrebbe dipendere da: quanto margine/vantaggio hai già (gap attuale rispetto al gruppo dietro), quanto manca alla fine (meno strada residua, meno tempo per essere recuperato — coerente con Chapatte's Law, che ti dà un punto di calibrazione empirico concreto: un minuto di vantaggio si erode in circa 10 km, o 90 secondi con le condizioni moderne).
python

def estimate_probability_being_caught(self):
    gap_time = self.gap_to_chasers / self.v  # vantaggio in tempo, non solo spazio
    time_remaining = self.distance_to_finish / self.v
    
    # quanto gap si "consuma" per km, calibrato da Chapatte's law
    erosion_rate = 1.0 / 10  # minuti di gap erosi per km, valore di letteratura
    
    gap_at_finish = gap_time - erosion_rate * self.distance_to_finish_km
    
    return sigmoid(-k_catch * gap_at_finish)  # gap_at_finish negativo → alta p_caught

Questo ti dà una formula con base empirica solida (Chapatte's law) invece di un parametro arbitrario, che è coerente con la tua preferenza di minimizzare parametri liberi non ancorati.

Questa unificazione ti sembra completa, o resta qualche caso limite scoperto (per esempio: cosa succede se il "gruppo davanti" è a sua volta in fuga e accelera mentre lo stai inseguendo — il target si muove, va ricalcolato ogni timestep, ma presumo sia automatico nel tuo loop)?

in questo modello cé accelerazione istantanea, quindi io calcolerei con la velocitá dellíntervallo. una cosa importante: dato che la probabilitá interna degli agenti nel valutare il numero di inseguiti in un breakaway é bayesiana, fa update. ma léffettivo numero di persone che entrano nel breakaway dipende a sua volta dalla distribuzione. non sono osservazioni indipendenti
1:15 AM

Hai messo il dito su un problema reale e sottile dell'inferenza bayesiana in questo contesto. Vediamolo con attenzione perché è importante per la validità statistica del meccanismo.
Il problema esatto

Quando un agente stima p_follow (probabilità che ciascun altro agente lo segua) e poi osserva l'esito (es. "in questo breakaway mi hanno seguito 3 persone su 10 potenziali"), vorrebbe trattare quell'osservazione come un campione da una binomiale con parametro p_follow fisso e N tentativi indipendenti.

Ma non è così, per il motivo che hai identificato: le decisioni dei follower non sono indipendenti tra loro. Se l'agente C osserva che B ha appena deciso di seguire, questo influenza la decisione di C (più gente nel breakaway → più conviene seguire, per il discorso che avevamo fatto su U_mean(k+1) che dipende dalla dimensione finale). Quindi gli eventi "segue o non segue" di agenti diversi sono correlati positivamente (effetto a cascata/contagio), non indipendenti come assume una binomiale standard.
Conseguenza pratica se ignori il problema

Se l'attaccante fa update bayesiano assumendo indipendenza (modello binomiale standard, prior Beta), la sua stima di p_follow sarà sistematicamente distorta. In particolare: osserverà spesso esiti "estremi" (tutti seguono, o nessuno segue) più spesso di quanto un vero processo binomiale indipendente predirebbe, perché la correlazione positiva tra le decisioni produce sovradispersione rispetto alla binomiale. Un modello binomiale standard sottostimerebbe la varianza vera, portando a un apprendimento bayesiano overconfident (l'agente diventa "troppo sicuro" troppo in fretta, aggiornando con un peso maggiore di quanto l'evidenza realmente giustifichi).
Soluzioni possibili, in ordine di complessità crescente

Opzione 1 — usa una distribuzione con sovradispersione invece della binomiale pura

La beta-binomiale è lo strumento standard per questo: invece di assumere p_follow fisso e noto, assumi che p_follow stesso vari (catturando l'effetto a cascata come "rumore extra" senza modellare esplicitamente la dipendenza) — è matematicamente equivalente a un modello binomiale con parametro p estratto da una Beta ogni volta, il che produce naturalmente più varianza nelle osservazioni di k. Questo è probabilmente la soluzione più pragmatica per te: stesso framework Beta-Binomial bayesiano che probabilmente già avevi in mente, ma accettando che la varianza osservata sarà più alta del previsto da una binomiale pura, senza dover modellare esplicitamente il meccanismo di cascata.

Opzione 2 — modella esplicitamente la cascata come processo sequenziale

Più fedele alla realtà ma più complesso: invece di "N agenti decidono simultaneamente e indipendentemente", modelli la decisione come sequenziale — ogni agente decide guardando quanti hanno già deciso di seguire prima di lui (questo è già implicito nel tuo modello se le decisioni avvengono nello stesso timestep ma in un ordine, anche se con SimultaneousActivation tutti decidono "alla cieca" sullo stato del timestep precedente, quindi la cascata avviene naturalmente tra timestep, non dentro lo stesso timestep).

Punto importante: con SimultaneousActivation (che già usi), la cascata non avviene istantaneamente entro un timestep — ogni agente decide guardando solo lo stato osservabile fino al timestep precedente. Quindi la correlazione tra decisioni si manifesta su più timestep consecutivi, non in un singolo istante. Questo in realtà attenua il problema che hai sollevato: dentro un singolo timestep, le decisioni sono effettivamente indipendenti condizionatamente allo stato osservato fino a quel punto (ogni agente vede lo stesso stato pubblico e decide per conto suo). La dipendenza emerge solo guardando la sequenza temporale completa (il numero finale di follower nel breakaway, accumulato su più timestep).
Quindi la domanda cruciale da chiarire: cosa sta facendo update l'agente esattamente?

Se l'attaccante aggiorna la sua credenza una volta a fine episodio (osservando il numero finale k di follower accumulati su tutta la durata del breakaway), allora sì, hai il problema di non-indipendenza che hai descritto — quel k finale è il risultato di un processo dinamico correlato nel tempo, non di N coin-flip indipendenti.

Se invece l'attaccante aggiorna timestep per timestep (ad ogni istante, osserva chi si aggiunge "ora" dato lo stato pubblico fino a "ora"), allora ogni singolo timestep è più vicino a un'osservazione condizionatamente indipendente (dato lo stato pubblico corrente, ogni agente che ancora non ha deciso decide per conto suo) — e il bayesian updating timestep-per-timestep è più giustificabile, anche se resta il fatto che lo stato pubblico stesso (quanti hanno già seguito) cambia il parametro effettivo nel tempo (non è più stazionario).

Consiglio pratico: tratta p_follow come non stazionario nel tempo all'interno dello stesso episodio di breakaway — non un singolo parametro fisso da stimare con N osservazioni indipendenti, ma una funzione che già include "quanti hanno seguito finora" come input osservabile:
python

p_follow_instantaneous(t) = sigmoid(f(distance_to_end, k_already_following(t), ...))

Questo evita il problema alla radice: non stai più trattando le decisioni come N tentativi i.i.d. di un singolo parametro nascosto — stai modellando esplicitamente che la probabilità cambia (aumenta, plausibilmente) man mano che più gente si aggrega, e l'agente osserva/aggiorna su questa dinamica, non su un singolo numero finale k trattato come binomiale pura.
Riformulazione consigliata

Invece di "l'attaccante stima un singolo p_follow e fa update bayesiano su k finale", fai: l'attaccante mantiene una credenza su come p_follow cresce con k_already_following (l'effetto cascata stesso diventa l'oggetto della stima, non un parametro nascosto fisso). Questo è più complesso da formalizzare ma è l'unico modo onesto di rispettare la non-indipendenza che hai correttamente identificato — altrimenti il bayesian updating, anche se concettualmente presente, si baserebbe su un'assunzione statisticamente sbagliata (i.i.d.) che un revisore esperto potrebbe contestare.

Vuoi che proviamo a formalizzare questa versione "dinamica" di p_follow (dipendente da k_already_following osservato in tempo reale), magari mantenendo comunque un singolo parametro libero da stimare per semplicità?

potresti fare un riassunto di tutto quello che abbiamo detto di questúltimo modello, con le cose che reputi migliori?
9:02 AM
Riassunto del modello — versione "utility-mediated interaction"
Struttura generale

Ogni ciclista è un agente discreto su una retta 1D, con stato interno (posizione, velocità, energia W', credenze bayesiane) e parametri fisici fissi (CP/Wmax derivati da un singolo parametro, team_id). I gruppi non sono entità decisionali — emergono come misurazione di prossimità (<3m) da decisioni individuali, mai assegnati o decisi collettivamente. Questo rispetta il requisito di assenza di controllo centrale.
Le azioni fondamentali (unificate)

Hai convergentemente ridotto tutto a un piccolo numero di meccanismi generici, riusati in contesti diversi:

1. Muoversi verso un target. Ogni agente isolato ha sempre un target: il prossimo gruppo/agente davanti, oppure — se è già in testa — il traguardo stesso. "Attaccare", "inseguire un breakaway", "raggiungere il gruppo davanti" sono tutti casi dello stesso meccanismo, con target diverso. Riduce drasticamente il numero di formule distinte necessarie.

2. Reagire a chi si avvicina. Quando un agente B viene raggiunto da A, B decide se scappare o tollerare la fusione — usando esattamente la stessa struttura di utility del "muoversi verso un target" (p_sustain, U_mean(M), costo energetico), solo innescata reattivamente. Nessun meccanismo nuovo: stesso framework, trigger diverso.

3. Cooperare/defezionare nel gruppo. Threshold public goods game: ogni agente decide uno sforzo continuo c_i ∈[0,1], stimando quanto lavoro serve per un obiettivo (es. raggiungere un gruppo davanti) e quanto è già stato fatto — inferito da segnali osservabili (velocità aggregata), non da conoscenza diretta dello sforzo altrui. La velocità del gruppo emerge come media pesata sui contributi del timestep precedente (rompe la circolarità, stesso pattern step/advance già usato per la fisica).
Elementi che ritengo più solidi

La velocità sostenibile v_s come cubica risolvibile in forma chiusa (stessa tecnica di Cardano già usata per aerobic_speed): dato energia residua e distanza, trova la velocità costante che esaurisce esattamente W' al traguardo. Riusata ovunque serve valutare "posso sostenere questo ritmo fino alla fine?".

Chapatte's Law come ancoraggio empirico per stimare p_caught (probabilità di essere ripreso da dietro): un vantaggio si erode a un ritmo noto (~10km o 90 secondi per minuto di gap), dato dalla letteratura — non un parametro arbitrario.

Il bonus di team modulato, non additivo: seguire un compagno vale p_win_compagno_stimata · bonus_team, non un bonus fisso — così l'incentivo a sacrificarsi scala con quanto è realisticamente probabile che il compagno vinca.

Separazione netta tra ciò che è ambientale/fisso e ciò che è strategico/evolve: parametri come reduction_max, N_scale, l'erosione di Chapatte sono dati noti dalla letteratura, fuori dalla sensitivity analysis dei 5 parametri principali. Le strategie individuali (λ_cost, bias di rischio, parametri di soglia) sono ciò che il replicator dynamics fa evolvere.
Il problema aperto più importante, non ancora risolto

L'inferenza bayesiana su p_follow viola l'indipendenza. Hai identificato correttamente che il numero finale di follower in un breakaway non è una somma di N tentativi i.i.d. — c'è un effetto a cascata (vedere altri seguire aumenta la probabilità che io segua). La soluzione più onesta è trattare p_follow come dipendente dal numero di follower già osservato in quel momento (p_follow(k_already_following, distance)), invece di stimare un singolo parametro fisso nascosto con update bayesiano su osservazioni trattate come indipendenti. Con SimultaneousActivation il problema si attenua entro un singolo timestep ma resta vero sull'intero episodio — è il pezzo matematicamente più delicato ancora da formalizzare del tutto.
Requisiti di progetto — stato attuale

Soddisfatti con buona sicurezza: entità discrete, stati interni, spazialità, percezione locale, bounded rationality, apprendimento (doppio livello: bayesiano dentro la gara + replicator dynamics tra gare), struttura game-theoretic formale (public goods game con free-riding, confermato anche dalla letteratura empirica che hai trovato).

Ancora scoperto esplicitamente: risk aversion e loss aversion (punto 6) — discussi concettualmente (penalità su varianza, utility asimmetrica alla Kahneman-Tversky) ma non ancora reintegrati nelle formule più recenti di attacco/seguire/cooperare. Da chiudere prima di considerare il modello completo.

Da verificare solo a posteriori, eseguendo il modello: emergenza non triviale (differenziazione leader/domestique da parametri fisici, timing di attacco apprenso) — il design lo rende plausibile ma va dimostrato sui risultati, non dichiarato a priori.

in effetti si potrebbe fare che l'agente che segue un attaco lo fa sulla base di sustain_probability, per vedere se riuscirebbe a sostenere quella velocitá. chi attacca allora puó avere la sua probabilitá di essere inseguito che cambia nel tempo, con bayes.
10:44 AM

Buona soluzione, e risolve elegantemente il problema di non-indipendenza che avevamo identificato. Vediamo perché funziona meglio.
Perché questo risolve il problema

Se ogni potenziale follower decide solo in base alla propria p_sustain (può io, con la mia energia residua e distanza, sostenere questa velocità fino alla fine?), allora la decisione di ciascun agente non dipende più da cosa fanno gli altri — dipende solo dal proprio stato interno (energia, CP) confrontato con la velocità osservata dell'attaccante. Questo è genuinamente indipendente tra agenti (condizionatamente alla velocità dell'attaccante, che è pubblica), perché non c'è più l'effetto cascata "vedo che altri seguono, quindi seguo anche io" — ogni agente valuta solo se è fisicamente capace, non cosa fanno gli altri.

Questo elimina la correlazione che rendeva problematico il bayesian updating sul numero di follower, perché ora k_follower è davvero una somma di eventi condizionatamente indipendenti dato lo stato pubblico (velocità dell'attaccante).
Cosa cambia per l'attaccante

L'attaccante ora osserva, episodio dopo episodio (o nel corso della stessa gara, attacchi multipli), quanti agenti hanno una p_sustain sufficiente da seguirlo a una data velocità — e fa update bayesiano su quanto è probabile che gli altri possano sostenere il suo ritmo, che è una quantità più stabile e interpretabile di "quanti mi seguiranno per ragioni sociali/strategiche".
python

# l'attaccante stima, per ogni potenziale follower osservato, 
# se ha probabilità alta o bassa di sostenere v_attacco
# aggregando su tutti i potenziali follower nel gruppo

p_follow_estimated = aggregate(p_sustain_i for i in observed_group)

E l'update bayesiano ora è su un parametro più sensato: quanto sono "forti" in media gli agenti nel gruppo che sto lasciando, una proprietà che varia gara per gara (composizione del gruppo) ma che è coerente con un modello binomiale, perché ogni p_sustain_i è valutato indipendentemente.
Un dettaglio da chiudere: l'attaccante non conosce i p_sustain altrui esattamente

L'attaccante non sa con certezza chi può sostenere il suo ritmo (non conosce l'energia interna altrui) — quindi la sua stima di p_follow deve restare una stima aggregata e rumorosa, aggiornata via bayes osservando gli esiti (quanti effettivamente lo seguono), non i singoli p_sustain_i (che restano privati). Questo è coerente con quanto avevi già — l'attaccante mantiene un prior su "quanto è probabile che la gente intorno a me possa sostenere attacchi di questa intensità", e lo aggiorna con l'evidenza osservata (k effettivo), che ora è statisticamente più solida perché ogni singola decisione di seguire (anche se non osservabile individualmente nel suo processo, solo nell'esito) è generata da un processo davvero più vicino all'indipendenza condizionale.
Riassumendo la correzione

Follower: decide se seguire SOLO in base a p_sustain_i(propria_energia, v_attacco, distanza)
          → decisione individuale, basata su stato proprio, non su cosa fanno gli altri

Attaccante: osserva k = numero di follower effettivi
            aggiorna (bayes) la propria credenza su "quanto è probabile che ad ogni livello 
            di intensità d'attacco, gli altri ce la facciano" 
            → ora l'update è su una binomiale con eventi più genuinamente indipendenti

Questo è un miglioramento netto rispetto a prima — più semplice da implementare (ogni follower fa un solo calcolo locale su sé stesso), più solido statisticamente per l'attaccante, e mantiene comunque l'incertezza/apprendimento bayesiano dove serve davvero (l'attaccante non può conoscere i p_sustain altrui, quindi impara dall'esperienza). Buona soluzione.

secondo me invece, ognuno che se ne va é una possibilitá per chi gli sta vicino di seguirlo
10:46 AM

Esatto, e questa è un'aggiunta importante che chiude un altro buco del modello. Vediamo cosa cambia.
Cosa stavi assumendo implicitamente prima, e perché era incompleto

Nella formulazione precedente, sembrava che ci fosse "un attaccante" e poi "i potenziali follower" decidono in base a p_sustain rispetto a quell'unico attaccante. Ma nella realtà — e nel tuo modello, se è coerente — ogni volta che qualcuno si stacca dal gruppo (non solo "l'attaccante designato", ma chiunque, incluso un follower che a sua volta decide di proseguire da solo, o un secondo agente che reagisce al primo), questo crea un nuovo evento osservabile che chiunque sia nelle vicinanze può valutare.

Questo significa che il processo non è "un attacco, poi N decisioni simultanee" — è una cascata di eventi sequenziali, dove ogni distacco genera nuove opportunità di follow per chi è vicino in quel momento.
Come si traduce questo nel modello

timestep t:   agente X si stacca dal gruppo principale (decisione individuale di X)
timestep t+1: agenti vicini a X (non necessariamente tutti quelli del gruppo originale) 
              osservano X e valutano p_sustain rispetto alla sua velocità
              → alcuni decidono di seguire X
timestep t+1: SE qualcuno segue X, si forma un piccolo gruppo (X + follower)
              gli agenti CHE NON erano abbastanza vicini a X per accorgersene in tempo,
              o che hanno deciso di non seguire, restano nel gruppo principale
timestep t+2: ora il gruppo (X + follower) PUÒ ESSERE A SUA VOLTA un target per altri 
              agenti ancora più indietro, che vedono questo nuovo gruppo allontanarsi
              e valutano se inseguirlo (stesso meccanismo "raggiungi il gruppo davanti")

Questo è coerente con quanto avevi già — "muoversi verso un target" si applica già a chiunque veda un gruppo/agente davanti a sé allontanarsi, indipendentemente da chi fosse "l'attaccante originale". Quindi in realtà non serve un meccanismo nuovo: il meccanismo che già avevi (target = prossimo gruppo davanti) gestisce naturalmente questa cascata, a patto che venga rivalutato ogni timestep da ogni agente, non solo "una volta, in risposta all'attacco iniziale".
La cosa importante che cambia: la simultaneità con cui valuti la "popolazione di potenziali follower"

Se prima pensavi a "l'attaccante + il resto del gruppo originale, che decidono tutti insieme", ora il modello corretto è: ad ogni timestep, ogni agente isolato o in un gruppo guarda chi ha davanti (che potrebbe essere l'attaccante originale, o un gruppo che si è già formato un timestep prima, o un secondo agente che si è staccato indipendentemente) e decide se inseguire, usando sempre lo stesso identico criterio (p_sustain rispetto alla velocità osservata del target).

Questo significa che il numero di follower di un dato "movimento" non è deciso in un singolo istante — emerge come processo dinamico su più timestep, con la possibilità che il gruppo che fugge cresca progressivamente (man mano che altri, uno dopo l'altro, decidono indipendentemente di aggiungersi) o che si formino più sottogruppi distinti (alcuni seguono X, altri restano, altri ancora seguono un terzo agente Y che si è staccato per conto suo nello stesso periodo).
Conseguenza positiva: hai emergenza genuina nella dinamica dei breakaway

Questo è in realtà un buon esempio di emergenza non banale (punto 10): la dimensione e composizione finale di un gruppo di fuga non è decisa da nessuno, nemmeno dall'attaccante iniziale — emerge dalla somma di tante decisioni locali e sequenziali, ognuna basata solo su informazione locale (chi vedo davanti, posso sostenere il suo ritmo). Pattern complessi come "un attacco che inizialmente aveva 2 follower, ne guadagna altri 3 dopo qualche minuto perché il gap si è stabilizzato a una velocità sostenibile da altri" possono emergere naturalmente, senza che tu li abbia scritti come regola.
Cosa devi solo verificare nell'implementazione

Che il meccanismo "valuta il gruppo/agente davanti a te, decidi se inseguire" venga effettivamente ricalcolato ogni timestep per ogni agente, non solo triggherato una tantum quando "qualcuno attacca" — altrimenti perderesti la possibilità di cascata progressiva che hai appena descritto. Dato come hai strutturato il loop (step/advance, ogni agente valuta il proprio stato ogni timestep), questo dovrebbe già essere naturale nella tua architettura, a patto che get_target() (gruppo/agente davanti) venga richiamato ad ogni timestep e non solo "all'inizio dell'attacco".

Mi sembra che questo si integri bene con tutto quello che avevi già senza richiedere nuove formule — è più una questione di assicurarsi che il meccanismo esistente sia applicato in modo continuo nel tempo, non episodico. Sei d'accordo, o vedi un caso in cui serve qualcosa di esplicitamente nuovo?

class CyclistAgent(Agent):
    """A single rider. State only; the model drives the dynamics."""

    def __init__(self, model, team_id: int, coeffs: dict | None = None):
        super().__init__(model)
        self.team_id = team_id

        cfg = model.config
        # Heterogeneous engine; floor keeps the Gaussian tail physical.
        self.w_max10 = max(50.0, model.random.gauss(cfg.w_max10_mean, cfg.w_max10_std))
        # Derived physiology, all filled by init_physiology below.
        self.cp = self.s_m = self.s_cp = self.s_sustain = self.w_full = self.w_prime = (
            0.0
        )
        energy.init_physiology(self, cfg)

        self.coeffs = (
            coeffs if coeffs is not None else strategy.default_coeffs(model.random)
        )
        self.solo = False  # True while a rider is off the front / chasing a breakaway (cleared after cooldown)
        self.break_cooldown = (
            0  # short-term memory to keep recent breakers separate from original packs
        )
        self.exposure = 1.0  # cf_eff-derived wind exposure, for the viz
        self.utility = 0.0  # race-outcome score, read by evolution

def default_coeffs(rng: random.Random | None = None) -> dict:
    """Sample a fresh set of coefficients for one rider."""
    rng = rng or random.Random()
    return {
        group: {
            name: rng.gauss(
                DEFAULT_COEFF_MEANS[group][name], DEFAULT_COEFF_STDS[group][name]
            )
            for name in params
        }
        for group, params in DEFAULT_COEFF_MEANS.items()
    }


def sigmoid(z: float) -> float:
    # Branch by sign so math.exp never sees a large positive argument (overflow);
    # evolution can drive coefficients large, so this must stay stable everywhere.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _distance_frac(agent, cfg) -> float:
    """Remaining distance to the finish as a fraction of the course (1 -> 0)."""
    return max(0.0, (cfg.road_length - agent.pos[0]) / cfg.road_length)


def _energy_frac(agent) -> float:
    return agent.w_prime / agent.w_full if agent.w_full else 0.0


def _teammates_in(agent, group) -> int:
    return sum(1 for o in group if o is not agent and o.team_id == agent.team_id)


def contribution(agent, group, cfg) -> float:
    """C_i = sigma(alpha + beta*d/L + gamma*T + delta*(1-W'/W_full), in (0, 1)."""
    c = agent.coeffs["coop"]
    z = (
        c["alpha"]
        + c["beta"] * _distance_frac(agent, cfg)
        + c["gamma"] * _teammates_in(agent, group)
        + c["delta"] * (1 - _energy_frac(agent))
    )
    # print(f"Agent {agent.unique_id} contribution z: {z:.3f} ,", "sigmoid:", sigmoid(z))
    return sigmoid(z)


def sustain_probability(agent, v_group: float) -> float:
    a = agent.coeffs["attack"]
    return sigmoid(a[0] + (agent.s_sustain - v_group) * a[1])

10:55 AM

in agent.py:

"""The cyclist agent: physiological + strategic state.

Behaviour (grouping, drafting, pace, breakaways) is orchestrated pack-wise by
``PelotonModel.step`` — a single rider can't decide a pack's speed alone — so the
agent itself only holds state.
"""

from mesa import Agent

from peloton import energy, strategy


class CyclistAgent(Agent):
    """A single rider. State only; the model drives the dynamics."""

    def __init__(self, model, team_id: int, coeffs: dict | None = None):
        super().__init__(model)
        self.team_id = team_id

        cfg = model.config
        # Heterogeneous engine; floor keeps the Gaussian tail physical.
        self.w_max10 = max(50.0, model.random.gauss(cfg.w_max10_mean, cfg.w_max10_std))
        # Derived physiology, all filled by init_physiology below.
        self.cp = self.s_m = self.s_cp = self.s_sustain = self.w_full = self.w_prime = (
            0.0
        )
        energy.init_physiology(self, cfg)

        self.coeffs = (
            coeffs if coeffs is not None else strategy.default_coeffs(model.random)
        )
        self.solo = False  # True while a rider is off the front / chasing a breakaway (cleared after cooldown)
        self.break_cooldown = (
            0  # short-term memory to keep recent breakers separate from original packs
        )
        self.exposure = 1.0  # cf_eff-derived wind exposure, for the viz
        self.utility = 0.0  # race-outcome score, read by evolution

in strategy.py: 

def default_coeffs(rng: random.Random | None = None) -> dict:
    """Sample a fresh set of coefficients for one rider."""
    rng = rng or random.Random()
    return {
        group: {
            name: rng.gauss(
                DEFAULT_COEFF_MEANS[group][name], DEFAULT_COEFF_STDS[group][name]
            )
            for name in params
        }
        for group, params in DEFAULT_COEFF_MEANS.items()
    }


def sigmoid(z: float) -> float:
    # Branch by sign so math.exp never sees a large positive argument (overflow);
    # evolution can drive coefficients large, so this must stay stable everywhere.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _distance_frac(agent, cfg) -> float:
    """Remaining distance to the finish as a fraction of the course (1 -> 0)."""
    return max(0.0, (cfg.road_length - agent.pos[0]) / cfg.road_length)


def _energy_frac(agent) -> float:
    return agent.w_prime / agent.w_full if agent.w_full else 0.0


def _teammates_in(agent, group) -> int:
    return sum(1 for o in group if o is not agent and o.team_id == agent.team_id)


def contribution(agent, group, cfg) -> float:
    """C_i = sigma(alpha + beta*d/L + gamma*T + delta*(1-W'/W_full), in (0, 1)."""
    c = agent.coeffs["coop"]
    z = (
        c["alpha"]
        + c["beta"] * _distance_frac(agent, cfg)
        + c["gamma"] * _teammates_in(agent, group)
        + c["delta"] * (1 - _energy_frac(agent))
    )
    # print(f"Agent {agent.unique_id} contribution z: {z:.3f} ,", "sigmoid:", sigmoid(z))
    return sigmoid(z)


def sustain_probability(agent, v_group: float) -> float:
    a = agent.coeffs["attack"]
    return sigmoid(a[0] + (agent.s_sustain - v_group) * a[1])

vorrei mettere la probabilitá percepita di essere seguito. secondo te ha senso oppure il modello funziona lo stesso? ovvero la probabilitá p_follow(k) che discutevamo prima. con léffetto a cascata é un pó difficile da implementare, e poi forse l'agente puó imparare quando gli conviene attaccare da k_attack e il bias allínterno di p_sustain. Oppure é piú interessante fare update bayesiano?
10:58 AM

Buona domanda pratica — guardando il codice che hai già, c'è effettivamente un argomento forte per non introdurre p_follow esplicito, almeno per ora. Vediamo perché.
Cosa fa già il modello senza p_follow esplicito

Guardando sustain_probability, ogni agente vicino a un attaccante valuta autonomamente se può sostenere v_group (la velocità imposta dall'attaccante/breakaway) confrontandola con il proprio s_sustain. Questo è esattamente la soluzione "ognuno decide in base alla propria p_sustain" che avevamo concluso essere la più solida — ogni follower decide per sé, indipendentemente dagli altri, niente cascata da modellare esplicitamente nel meccanismo di follow.

Quindi: il modello funziona già correttamente senza p_follow — è l'attaccante che, se vuoi dargli capacità predittiva, avrebbe bisogno di p_follow per decidere se attaccare (calcolando il beneficio atteso pesato sul numero atteso di follower). Ma chi segue non ne ha bisogno: decide già con sustain_probability.
Quindi la domanda si restringe a: serve a chi attacca?

Sì, per una ragione specifica: senza qualche stima di "quanti mi seguiranno", l'agente che decide se attaccare non ha modo di valutare il trade-off corretto (attaccare da solo con pochi follower è ottimo se sei forte; attaccare quando tutti ti seguono vanifica lo sforzo). Senza p_follow, l'unica informazione che ha è la propria fisiologia (s_sustain) — può decidere "posso sostenere questo ritmo", ma non "vale la pena provarci dato chi mi sta intorno".
Le tue due alternative: k_attack/bias appreso vs update bayesiano — non sono mutuamente esclusive, sono complementari

k_attack e bias come parametri che evolvono (quello che hai già in coeffs["attack"]) catturano la propensione media dell'agente — quanto è "aggressivo" nell'attaccare, calibrato dall'evoluzione su tante gare. Questo cattura il pattern medio (es. "i ciclisti forti imparano soglie di attacco diverse dai deboli").

L'update bayesiano dentro la gara catturerebbe qualcosa di diverso e complementare: l'incertezza specifica di questa gara (questo gruppo è particolarmente combattivo o rilassato oggi?). Senza bayes, l'agente userebbe sempre lo stesso prior implicito (dato dai suoi coeffs evoluti), gara dopo gara, senza adattarsi alle condizioni specifiche dell'episodio corrente.
La mia raccomandazione pratica, vista la tua codebase

Dato che hai già una struttura pulita coeffs-per-categoria (coop, attack), e dato che il problema della cascata si risolve già grazie a sustain_probability (i follower non hanno bisogno di p_follow), ti consiglio:

Non implementare ora un vero update bayesiano (Beta-Binomial con tracking di stato persistente per ogni agente) — è complessità aggiuntiva non necessaria, dato che il problema di non-indipendenza che ti preoccupava è già risolto a monte dal meccanismo di sustain_probability.

Aggiungi invece un termine semplice e stimabile localmente nel calcolo di "vale la pena attaccare", basato su un proxy osservabile a costo quasi zero: la dimensione e composizione energetica stimata del gruppo attuale. Per esempio:
python

def expected_followers(agent, group, v_attack, cfg):
    """Stima quanti nel gruppo potrebbero sostenere v_attack — non bayes, solo aggregazione locale."""
    # l'agente non conosce w_prime altrui esattamente, ma può usare una stima 
    # basata su segnali pubblici: per esempio quanto sta "soffrendo" il gruppo
    # (proxy: velocità recente sostenuta vs velocità aerobica media)
    n_similar_strength = sum(
        1 for o in group if o is not agent and _looks_capable(o, v_attack)
    )
    return n_similar_strength

dove _looks_capable userebbe solo informazione pubblica plausibile (es. il rider è ancora nel gruppo principale invece di essere già stato sganciato in episodi precedenti — informazione che il tuo break_cooldown/storico già traccia implicitamente).
Risposta diretta alla tua domanda

Il modello funziona anche senza p_follow esplicito, perché la parte più delicata (la cascata tra follower) è già gestita bene da sustain_probability decisa individualmente. Aggiungere p_follow avrebbe senso solo per arricchire la decisione di attaccare, non per risolvere un problema esistente nel meccanismo di follow.

Se vuoi arricchirla, il bayesian update è concettualmente più elegante ma costa complessità implementativa (stato persistente per agente, gara dopo gara) per un beneficio che potrebbe essere marginale, dato che hai già l'evoluzione tra gare (coeffs["attack"]) che cattura il pattern medio. Ti consiglierei di partire senza bayes, verificare empiricamente se il modello produce comportamenti di attacco sensati usando solo sustain_probability + evoluzione dei coeffs, e aggiungere il bayesian update solo se osservi che manca qualcosa di specifico (es. gli agenti non si adattano a condizioni di gara molto diverse tra loro).

Vuoi che formalizziamo expected_followers o la decisione completa di "attacco sì/no" usando quello che hai già in coeffs["attack"], prima di considerare il bayes?

dovrei fare questo: 1. agenti guardano attorno. se vedono uno che va piú veloce di loro ed é vicino, calcolano p_sustain e inseguono, ovvero fanno uno sprint per arrivare vicino in un time step e poi cotninuano alla velocitá altrui. quando sono vicini, gli agenti hanno una riduzione della fatica contro l'aria data dal loro contributo C_i. oppure solo quando fanno parte dello stesso gruppo? ovvero, se uno raggiunge un altro, fanno parte dello stesso gruppo e si dividono il lavoro secondo effort c_i (che devo ancora decidere come implementare). inizio time step: gli agenti vedono se qualcuno va piú veloce di loro secondo una certo campo visivo e valutano se seguirlo o no. agenti che sono in gruppo decidono anche se volare verso il traguardo o no. update velocitá: in gruppo rimane uguale per chi non si stacca, cambia per chi vola; cambia per chi insegue qualcuno.  calcolo dei contributi nel gruppo per capire chi fa piú sforzo etc.
11:51 AM

Buona sequenza, abbastanza chiara. Rispondo prima alla domanda specifica, poi segnalo un paio di ambiguità nell'ordine delle operazioni.
Riduzione fatica: solo nello stesso gruppo, non per semplice vicinanza

Sì, dovrebbe essere solo quando fanno parte dello stesso gruppo (cioè quando un agente ha raggiunto l'altro, distanza sotto soglia, e quindi vengono misurati come stesso gruppo) — non per "vicinanza" generica durante l'inseguimento. Motivo: durante lo sprint per raggiungere qualcuno, l'agente sta ancora colmando il gap, non sta beneficiando della scia di chi sta inseguendo (fisicamente il draft richiede essere effettivamente dietro, a distanza di scia, non solo "nei dintorni"). Il draft/riduzione drag dovrebbe attivarsi dal timestep successivo a quello in cui il gap si chiude sotto soglia, non durante l'inseguimento stesso.

Questo è coerente con quanto hai già: _teammates_in e il calcolo di contribution operano già su group, quindi la riduzione del costo aerodinamico per draft dovrebbe applicarsi alla stessa lista di membri del gruppo già misurata, non a una nozione separata di "vicinanza visiva".
L'ambiguità nell'ordine: due decisioni nello stesso timestep rischiano di confondersi

Hai descritto: "inizio timestep, vedono se qualcuno va più veloce e valutano se seguirlo" — e separatamente "agenti in gruppo decidono anche se volare". Queste sono due decisioni potenzialmente in conflitto se non ordinate bene: un agente nel gruppo potrebbe contemporaneamente (a) vedere qualcuno fuori dal gruppo che va più veloce e valutare di inseguirlo, e (b) valutare se essere lui stesso a volare via dal gruppo. Sono la stessa famiglia di decisione (target = fuori dal gruppo) vista da prospettive diverse — rischio di doppio conteggio o codice che fa due calcoli ridondanti.

Consiglio: unifica in una sola valutazione per agente per timestep, con priorità chiara:

per ogni agente, in un singolo step decisionale:
  1. c'è qualcuno (gruppo o singolo) più avanti e più veloce, che il mio gruppo non sta già 
     raggiungendo collettivamente? → valuta insegui/non inseguire (usa sustain_probability)
  2. SE non sto inseguendo nessuno: valuta se voglio essere io a partire (volare) 
     → stessa logica, target = traguardo se sei già il gruppo più avanti, 
       altrimenti target = il prossimo gruppo che comunque hai appena valutato al punto 1

In pratica i punti 1 e 2 sono lo stesso identico calcolo (target = qualcosa avanti a me, decido se accelerare per raggiungerlo/anticiparlo) — non servono due decisioni separate, è una sola decisione "accelero o no", solo che il target cambia in base a cosa l'agente vede.
Sull'update di velocità — la tua descrizione è quasi giusta, manca un caso

Hai detto: "in gruppo rimane uguale per chi non si stacca, cambia per chi vola, cambia per chi insegue". Manca il caso di chi viene raggiunto (l'agente B di prima, che deve decidere se tollerare o scappare quando qualcuno lo sta per raggiungere da dietro) — se B decide di tollerare, la sua velocità potrebbe dover scendere/salire per allinearsi al nuovo gruppo che si forma con A. Se scappa, accelera. Verifica che questo quarto caso sia coperto nel tuo schema di update, non solo "chi vola" e "chi insegue" (che sono entrambi iniziativa propria), ma anche "chi reagisce passivamente a essere raggiunto".
Sul calcolo dei contributi — un punto tecnico sull'ordine

Hai messo "calcolo dei contributi nel gruppo" come ultimo passo. Verifica che usi la composizione del gruppo dopo che si sono risolte le partenze/arrivi di quel timestep (altrimenti calcoli contribution per un gruppo che sta per cambiare composizione, con _teammates_in potenzialmente già obsoleto). L'ordine corretto dovrebbe essere:

1. ogni agente valuta/decide azione (insegui, vola, resta) — basato su STATO INIZIO timestep
2. aggiorna posizioni/velocità in base alle decisioni (chi parte, chi insegue accelera)
3. ricalcola i gruppi (nuova misurazione di prossimità, dopo i movimenti)
4. SOLO ORA calcola contribution() per il NUOVO assetto di gruppi, per il prossimo timestep

Questo rispetta lo stesso pattern step/advance che già usi per evitare la circolarità — il contribution calcolato ora servirà per decidere il costo energetico/velocità del prossimo timestep, non di questo.

Ti sembra corretto questo ordinamento, o preferisci vedere lo pseudocodice completo di un singolo timestep per verificarlo riga per riga?

ma per calcolare contribution serve sapere chi resta nel gruppo. se calcolo il contributo alla fine, per il prossimo timestep, magari qualcuno che avevo calcolato presente poi se ne va e vola
11:57 AM

Hai ragione, c'è effettivamente un problema di sincronizzazione — ma è risolvibile, e in realtà è lo stesso tipo di problema (decisione basata su stato leggermente vecchio) che hai già accettato altrove nel modello con SimultaneousActivation. Vediamo la soluzione corretta.
Il problema preciso

Se calcoli contribution() a fine timestep t per il gruppo G(t), e poi al timestep t+1 qualcuno decide di volare via, il contribution calcolato per quell'agente (e di conseguenza la velocità di gruppo derivata) era basato su una composizione che non esiste più. Risultato: per un timestep, il gruppo "pensa" di avere ancora quel membro per calcolare la velocità, ma fisicamente non c'è più.
Perché questo è in realtà accettabile (e già presente nel tuo modello)

Con SimultaneousActivation, ogni decisione è già basata su uno stato che è "un timestep vecchio" rispetto al momento in cui viene eseguita — è il prezzo che paghi per evitare dipendenze circolari e bias di ordine di attivazione. Non è un bug, è una caratteristica intrinseca e accettata del paradigma. La domanda è solo: quanto stale può essere lo stato prima che diventi un problema reale (numericamente o concettualmente)?

Per un singolo timestep (se il dt è piccolo, es. pochi secondi), l'errore introdotto da "calcolare la velocità di gruppo con un membro che sta per partire" è marginale — quell'agente contribuirà alla velocità di gruppo per un istante in più di quanto dovrebbe fisicamente, poi al timestep successivo il gruppo si ricalcola correttamente senza di lui. Non si accumula errore sistematico, è rumore locale che si autocorregge.
La sequenza corretta, che risolve l'ambiguità

Il punto chiave: il contributo che entra nel calcolo della velocità del gruppo per il timestep corrente deve essere quello calcolato alla fine del timestep precedente, quando il gruppo era ancora "stabile" per quella decisione. Quello che cambia ora (chi vola via) è una nuova decisione presa con il nuovo timestep, che si applicherà al gruppo dal prossimo step in poi.

TIMESTEP t:
  1. USA contribution(t-1) per calcolare v_group(t) — composizione del gruppo COM'ERA a fine t-1
  2. Applica v_group(t) a tutti i membri ANCORA presenti nel gruppo (quelli che non sono già partiti)
  3. Ogni agente (incluso chi è nel gruppo) DECIDE ora, guardando lo stato corrente, 
     se vuole volare via nel prossimo istante
  4. Applica le decisioni: chi ha deciso di volare accelera, comincia a staccarsi
  5. Ricalcola i gruppi per prossimità (con le nuove posizioni)
  6. Calcola contribution(t) per il NUOVO assetto di gruppo — questo verrà usato al timestep t+1

Il punto cruciale: chi decide di volare al timestep t, contribuisce comunque (per quell'istante) alla velocità calcolata con i vecchi contributi, ma da quel preciso istante la sua decisione di staccarsi lo porta fuori dal gruppo per il calcolo successivo. Non c'è vera contraddizione — è semplicemente che l'effetto della sua partenza si manifesta a partire dal prossimo timestep, non istantaneamente nello stesso istante in cui decide.
Perché questo non è un problema serio nella pratica

Se il tuo dt è ragionevolmente piccolo (qualche secondo, dato che stai modellando una dinamica continua discretizzata), l'errore di "un timestep di ritardo" sulla composizione del gruppo è trascurabile rispetto alla dinamica complessiva della gara (che dura ore). È lo stesso tipo di approssimazione che fai already con W' balance, con la fisica del drag — sono tutte discretizzazioni di processi continui, e un timestep di lag su un parametro che cambia raramente (la composizione del gruppo non cambia ogni istante, cambia episodicamente) è un'approssimazione più che accettabile.
L'unica cosa da evitare

Non calcolare contribution() due volte nello stesso timestep con composizioni diverse (prima con il vecchio gruppo, poi ricalcolandolo dopo che qualcuno è partito, nello stesso ciclo) — questo causerebbe il vero problema di incoerenza che temevi. La regola pulita: un solo calcolo di contribution per timestep, fatto a fine ciclo con la composizione finale di quel timestep, usato per il timestep successivo. Mai ricalcolato "a metà" dello stesso ciclo per correggere decisioni appena prese.

Questo risolve il dubbio, o preferisci una versione alternativa in cui il contributo viene deciso "prospetticamente" (l'agente che sta per volare comunica/segnala la sua intenzione prima che il gruppo calcoli i contributi, evitando anche il lag di un timestep)? Quest'ultima sarebbe più precisa ma introduce un ordine di operazioni più delicato da implementare correttamente con Mesa.

un altro problema: per ora gli agenti non possono rallentare. se partono tutti con le velocitá che pensano di poter sostenere, semplicmenete il piú veloce non si aggrega ma si aggregano tutti gli altri oppure si aggregano tutti ma alla velocitá settata dal piú veloce
12:01 PM

Hai trovato un vero problema strutturale — senza la possibilità di rallentare, la dinamica collassa in modo degenere. Analizziamolo.
Perché collassa esattamente

Se ogni agente parte già alla propria velocità "ottimale" individuale (v_s, quella sostenibile dato energia e distanza), e non possono mai scendere sotto quella, allora quando si formano gruppi, l'unica velocità di gruppo fisicamente coerente è quella del più lento del gruppo (perché nessuno può andare più piano della propria scelta individuale, quindi il gruppo si "allarga" fino a includere tutti quelli che possono sostenere almeno quella velocità minima) — oppure, se la velocità di gruppo è decisa come media pesata o dal leader (più veloce), chi è più lento del valore imposto non può starci dentro per definizione (lui non rallenta mai sotto la sua v_s, quindi se il gruppo va più veloce di quanto lui può sostenere, si stacca immediatamente, non perché "decide" ma perché fisicamente non riesce a tenerlo).

Risultato degenere: o il gruppo si forma solo tra agenti con v_s simile (segmentazione automatica per livello fisico, che in realtà potrebbe non essere un problema — anzi), oppure, se forzi tutti alla velocità del più veloce, quelli più lenti vengono "expulsi" immediatamente sempre, ogni volta, rendendo i gruppi instabili o triviali.
La causa profonda: manca un incentivo a NON andare alla propria v_s ottimale

Il vero problema concettuale è che hai eliminato la possibilità che un agente scelga deliberatamente una velocità diversa dalla propria v_s ottimale individuale — ma nella realtà, un ciclista rallenta sotto la sua velocità sostenibile massima proprio per risparmiare energia stando in un gruppo più lento, accettando un compromesso. Questo è esattamente il senso del draft: vale la pena andare più piano di quanto potresti da solo, perché il gruppo ti permette di farlo consumando meno energia, lasciandoti più riserve per dopo (es. per lo sprint finale).
La soluzione: ridefinire v_s come tetto, non come velocità fissa

Il problema nasce dal trattare v_s come "la velocità a cui vado", invece che come il limite massimo sostenibile. Un agente razionale dovrebbe poter scegliere qualsiasi velocità ≤ v_s, e la sua decisione strategica è esattamente quale velocità scegliere in quel range, bilanciando:

v_alta (vicino a v_s):  arrivo prima se da solo, ma consumo di più, meno margine per dopo
v_bassa (sotto v_s):    risparmio energia (specialmente se in gruppo, con draft), 
                        ma se troppo bassa, rinuncio a competere per la posizione

Questo significa che stare in gruppo a una velocità inferiore alla propria v_s individuale è razionale, perché il costo energetico per quella velocità più bassa è ulteriormente ridotto dal draft — quindi anche se "potrei andare più forte da solo", mi conviene andare più piano insieme ad altri se il risparmio energetico netto (velocità più bassa + draft) supera il costo di perdere terreno nel breve termine.
Come si traduce questo nel modello pratico

L'agente, quando decide se restare nel gruppo o partire, dovrebbe confrontare:
python

U_stay_in_group = (beneficio di restare a v_group, con draft, risparmio energetico) 
U_leave_alone = (beneficio di andare a v_s_propria, da solo, senza draft, più energia consumata)

Questo è già esattamente la struttura di utility che avevamo costruito (U_mean(M) per restare, vs il calcolo di attacco per partire) — il punto che ti era scappato è che v_group può essere, e tipicamente sarà, inferiore a v_s individuale per gli agenti più forti, e questo è normale e atteso: gli agenti forti scelgono razionalmente di andare più piano di quanto potrebbero, perché il guadagno di gruppo (risparmio energia, posizionamento sociale, possibilità di sfruttare i compagni) supera il guadagno di andare da soli al proprio massimo.
Cosa ti manca quindi, concretamente

Non serve che gli agenti "rallentino" in un senso dinamico complesso — serve semplicemente che v_group non sia vincolata a essere uguale a v_s di qualcuno specifico, ma sia una variabile scelta collettivamente (emergente dalla somma dei contributi, come avevi già con la media pesata su contribution), che può essere strutturalmente più bassa della v_s del membro più forte, senza che questo sia un problema — è semplicemente il prezzo che un agente forte accetta di pagare (andare più piano di quanto potrebbe) in cambio dei benefici del gruppo.

La tua formula di velocità di gruppo come media pesata sui contributi già permette questo, a patto che tu non forzi v_group = v_s del leader (che era la formulazione di molto tempo fa, già abbandonata) — con la formula attuale (media pesata, dove ogni agente contribuisce c_i e la velocità risultante è una funzione di questi contributi, non semplicemente "quella del più forte"), il gruppo può naturalmente assestarsi su una velocità più bassa della v_s del più forte, perché è una media/aggregazione, non un massimo.
Verifica pratica che ti consiglio

Il problema che hai descritto ("il più veloce non si aggrega, o si aggregano tutti alla sua velocità") nasce probabilmente da un'implementazione attuale in cui la velocità di gruppo è ancora legata troppo rigidamente alla v_s di un singolo membro (es. il leader), invece di essere la media pesata sui contributi come avevi concettualmente già deciso. Controlla la formula esatta che hai implementato per v_group — se è ancora v_group = f(v_s_leader), va sostituita con qualcosa come:
python

def group_velocity(group, cfg):
    weights = [contribution(o, group, cfg) for o in group]
    speeds = [o.s_sustain for o in group]
    return weighted_average(speeds, weights) * k_s  # k_s ∈ [0.7, 1] come discusso

Questo permette naturalmente che v_group sia inferiore alla v_s del membro più forte, risolvendo il collasso che hai descritto.

Vuoi che verifichiamo insieme la formula esatta di group_velocity che hai nel codice attuale per capire se è quella la causa del problema?

magari posso fare p_sustain anche per chi va piú lento di me: quindi guardo chi mi sta vicino e valuto, se la sua velocitá é vicina alla mia mi aggancio, poi facciamo velocitá che é la media pesata
12:04 PM

Sì, ha senso e risolve il problema in modo elegante — diamo una struttura precisa.
L'idea corretta: simmetrizzare p_sustain, non solo per "raggiungere chi è più veloce"

Finora p_sustain rispondeva solo alla domanda "posso tenere il ritmo di chi è più veloce di me?". Ora estendi la stessa logica al caso opposto: "vale la pena rallentare per agganciarmi a chi è più lento di me?" — non è più solo una questione fisica (posso farlo, sì, rallentare è sempre fisicamente possibile), ma una questione di convenienza strategica.

Quindi il criterio cambia leggermente tra i due casi:

Per chi è più veloce: la domanda è "posso fisicamente sostenerlo?" → p_sustain come avevi.

Per chi è più lento: la domanda è "mi conviene rallentare per stare con lui?" → non è più una probabilità di sostenibilità fisica (rallentare è sempre sostenibile), è un confronto di utility: risparmio energetico del draft/gruppo vs costo di perdere tempo andando più piano di quanto potrei.
Formalizzazione pratica
python

def evaluate_neighbor(agent, other, v_self, v_other, group_candidate, cfg):
    if v_other > v_self:
        # caso "inseguo chi è più veloce" - già hai questo
        p = sustain_probability(agent, v_other)
        return p > threshold  # decido se tentare l'aggancio
    
    else:
        # caso "rallento per chi è più lento" - nuovo caso
        v_joint = weighted_average_speed(agent, other, group_candidate, cfg)
        
        u_join = utility_group(agent, v_joint, group_candidate, cfg)   # con draft
        u_alone = utility_solo(agent, v_self, cfg)                     # senza draft
        
        return u_join > u_alone

Questo evita di trattare "rallentare" come un problema di sostenibilità fisica (che non lo è) e lo tratta correttamente come una scelta economica/strategica — coerente con tutto il framework di utility che hai già costruito.
Sul "se la velocità è vicina alla mia mi aggancio" — buona euristica, ma chiarisci la soglia

Questo è un buon criterio pratico, semplice da implementare, ed è coerente con bounded rationality (l'agente non fa un calcolo complesso di utility ogni volta, usa un'euristica locale: "se è abbastanza simile a me, vale la pena provare ad allinearmi"). Solo una cosa da decidere: questa soglia di "vicinanza" è:

    Un parametro fisso del modello (es. "se la differenza di velocità è sotto 0.5 m/s, mi aggancio sempre") — semplice, ma è un nuovo parametro libero che si aggiunge ai tuoi 5 già allocati
    Derivata dal calcolo di utility sopra — non serve una soglia esplicita separata, l'agente fa il confronto u_join > u_alone e quello implicitamente determina "quanto vicina deve essere la velocità per valere la pena": più vicina è, meno costa rallentare, più facile che u_join vinca

Ti consiglio la seconda strada — non introduce un parametro nuovo, ed è più coerente: la "soglia" emerge naturalmente dal trade-off costo/beneficio già formalizzato, invece di essere un cutoff arbitrario.
Conseguenza interessante: ora la velocità di gruppo emerge da un vero negoziato implicito

Con questa estensione, il meccanismo diventa più completo: un gruppo si forma e si ri-assesta dinamicamente non solo per chi insegue attivamente, ma anche perché agenti più forti scelgono di rallentare leggermente per restare con un gruppo quasi alla loro andatura, se il guadagno energetico supera la piccola perdita di velocità. Questo è esattamente il meccanismo che mancava per evitare il collasso che avevi notato — ora la velocità di gruppo non è vincolata a essere quella del più forte o del più debole, ma è il risultato di scelte di "quanto mi conviene allinearmi" fatte da ciascun agente, in entrambe le direzioni (più veloce → valuta se può farcela; più lento → valuta se conviene).
Una semplificazione che ti consiglio per non duplicare logica

Nota che entrambi i casi (inseguire chi è più veloce, agganciarsi a chi è più lento) sono in fondo lo stesso confronto di utility — anche nel caso "più veloce", potresti formalizzarlo come u_join_inseguendo vs u_alone, con p_sustain che entra come fattore di sconto sul beneficio (perché c'è il rischio di non farcela), piuttosto che come gate binario separato:
python

def utility_join(agent, other, v_target, group_candidate, cfg):
    p_success = sustain_probability(agent, v_target) if v_target > agent.s_sustain else 1.0
    u_if_joined = utility_group(agent, v_target, group_candidate, cfg)
    u_if_fail = utility_solo(agent, agent.s_sustain, cfg)  # se non riesce a sostenerlo, resta solo
    
    return p_success * u_if_joined + (1 - p_success) * u_if_fail

Questo unifica i due casi in una sola formula coerente, dove p_success = 1.0 quando il target è più lento (rallentare è sempre possibile), e p_success = sustain_probability(...) quando il target è più veloce (incertezza fisica reale). Meno codice duplicato, stessa logica sottostante in entrambe le direzioni.

Ti sembra una buona unificazione? Vuoi che definiamo utility_group e utility_solo esplicitamente, magari riusando U_mean(M) e il costo energetico che avevi già?

ma come definisco utility if joined? solitamente per l'utility usavo la posizione in classifica attesa, ma qua non mi sembra il caso
12:08 PM

Hai ragione a fermarti qui — usare direttamente "posizione in classifica attesa" per una decisione così locale e frequente (valutare ogni vicino, ogni timestep) è sproporzionato e probabilmente fuorviante. Vediamo perché e cosa usare invece.
Perché la posizione in classifica non è la metrica giusta qui

La posizione finale attesa è una metrica a lungo orizzonte, sensata per decisioni come "attacco per la vittoria" o "resto nel gruppo principale" — situazioni con un legame relativamente diretto con l'esito di gara. Ma "mi aggancio a questo vicino che va leggermente più piano di me" è una micro-decisione tattica locale, il cui effetto sulla classifica finale è troppo indiretto e rumoroso da stimare in modo sensato a questo livello di granularità. Se provi a calcolare "che impatto ha questo specifico aggancio sulla mia posizione finale", il segnale si perde nel rumore — è come chiedere "che effetto ha questo singolo respiro sulla mia velocità media della gara".
La metrica giusta: risparmio energetico netto, nell'orizzonte immediato

Per decisioni locali di questo tipo, la metrica naturale è quella che già usi per tutto il resto del modello a livello "fisico": energia risparmiata/spesa, non posizione in classifica. La logica: l'energia che risparmi ora (o spendi) si traduce automaticamente in più opzioni strategiche dopo (puoi attaccare più tardi, sostenere meglio un breakaway, ecc.) — la classifica finale è già "tracciata" indirettamente, ma a un livello di decisione più alto, non in questa micro-scelta.
python

def utility_group(agent, v_target, group_candidate, cfg):
    """Beneficio energetico netto di unirsi al gruppo a v_target."""
    c = contribution(agent, group_candidate, cfg)  # quanto contribuirebbe
    p_air_with_draft = power_with_draft(agent, v_target, group_candidate, cfg)
    
    energy_cost_per_step = p_air_with_draft * cfg.dt
    
    return -energy_cost_per_step / agent.w_full   # normalizzato, negativo = costo

def utility_solo(agent, v_self, cfg):
    """Costo energetico di restare alla propria velocità, senza draft."""
    p_air_solo = power_solo(agent, v_self, cfg)
    energy_cost_per_step = p_air_solo * cfg.dt
    
    return -energy_cost_per_step / agent.w_full

In questo modo u_join > u_alone si traduce semplicemente in "il costo energetico per timestep, unendomi al gruppo a v_target, è inferiore al costo per andare da solo alla mia velocità" — che è esattamente il confronto economico naturale e locale, senza scomodare la classifica finale.
Ma manca un pezzo: il costo di velocità persa, non solo l'energia

Se consideri solo il risparmio energetico, un agente forte rallenterebbe sempre volentieri per un gruppo più lento (è sempre energeticamente vantaggioso andare piano), il che porterebbe tutti a convergere verso la velocità più bassa possibile — non corretto, perché ignora il costo opportunità di "arrivare più tardi". Serve un termine che penalizzi andare più piano della propria v_s, proporzionale a quanto tempo/distanza perdi:
python

def utility_join(agent, v_target, group_candidate, cfg):
    energy_benefit = -power_with_draft(agent, v_target, group_candidate, cfg) * cfg.dt / agent.w_full
    
    speed_penalty = lambda_speed * max(0, agent.s_sustain - v_target) / agent.s_sustain
    # penalizza andare più piano di quanto potresti, proporzionale al gap di velocità
    
    return energy_benefit - speed_penalty

Qui introduci un nuovo coefficiente lambda_speed — quanto pesa per l'agente "perdere tempo" rispetto a "risparmiare energia". Questo è esattamente il tipo di trade-off individuale che rende sensata la differenziazione leader/domestique: un agente con λ_speed alto preferisce sempre spingere al massimo (vuole arrivare presto, gli importa meno l'energia), un agente con λ_speed basso preferisce risparmiare energia anche a costo di andare più piano (più "paziente", aspetta occasioni migliori più avanti).
Perché questo è meglio della classifica diretta

Hai ottenuto un criterio locale, coerente con bounded rationality (l'agente valuta solo l'effetto immediato, non simula l'intera gara restante), che però aggregato su tante decisioni nel tempo produce comunque comportamento sensato verso l'obiettivo finale (chi risparmia energia ora, ne ha di più per attaccare dopo — l'effetto sulla classifica emerge indirettamente, senza doverlo calcolare esplicitamente ad ogni microdecisione).

lambda_speed può essere un parametro che evolve (parte della strategia individuale, come λ_cost che avevi già) oppure preferisci derivarlo da qualcos'altro che hai già nel modello, per non aggiungere un ulteriore grado di libertà nell'evoluzione?
