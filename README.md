# alph_stable

J'ai executé le fichier ipynb Test_notebook en prenant comme theta et gamma ce que j'ai appelé le cas 1 dans mon fichier theta, ce cas est celui où le maillage éloigne le plus les sources. Il donne les meilleurs résultats que j'ai pu avoir.

Le fichier Bss_evaluators ne sert pas j'avais mal compris comment estimer les BSS évaluateurs, je vais en faire un fichier jupyter cela sera plus pratique.

Le soucis est que je ne peux pas réaliser de STFT inverse sur le cluster car librosa a l'air de ne pas être compatible avec d'autres librairies de mon environnement virtuel.

Vous trouverez les audios des différents tests réalisés dans le dossier ./data/audio/in
Les audios sont enregistrés de la manière suivante: 
    -Si il y a le nom oracle mais que le nombre d'itérations est non nul,cela signifie que la partie NMF de la M-step est réalisée, c'est en fait semi oracle. Sinon c'est bien oracle.

    -M est le nombre de microphones, N le nombre de sources à séparer, K le nombre de bases pour la partie NMF (l'espace latent).

    -Si il y a init dans le nom, cela signifie que les matrices W et H de la NMF ont été initialisées par 100 itérations de l'algorithme FastMNMF2.

Pour l'instant tous les essais ont été réaliser avec alpha = 1.8 .

Si vous voulez faire les tests, vous pouvez utiliser le fichier shell en remplaçant dans celui ci mon identifiant jdpascal par le votre. Ce fichier shell vous connectera sur un noeud disponible du cluster, un lien vous sera envoyé pour communiquer avec l'interface jupyter sur votre navigateur. (Merci à Florian pour cet outil)