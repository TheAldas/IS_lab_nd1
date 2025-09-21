Uzduotis klasifikuoti maista.
Uzduociai atlikti pasirinktas https://huggingface.co/Jacques7103/Food-Recognition modelis veikiantis su PyTorch

Modeliui paleisti reikejo atsisiusti transformers ir torch bibliotekas
```
pip install transformers
pip install torch
```

ViTmodel dokumentacijoje nurodytas toks pavyzdinis kodas kaip paleisti ir gauti tekstinius label'ius:
<img width="600" alt="image" src="https://github.com/user-attachments/assets/b05cbcf2-eed3-4281-981b-e20303a79027" />


Pirminis kodas su pavyzdinio kodo modifikacijomis ir testavimui pasirinkta nuotrauka atrodo taip: 
<img width="600" alt="image" src="https://github.com/user-attachments/assets/4377398e-2504-46f2-b289-d500486c8503" />

Testavimui naudota nuotrauka:

<img width="275" height="183" alt="image" src="https://github.com/user-attachments/assets/233f0a60-8881-43ef-b3a0-13c68d35fab1" />

Rezultatai:
<img width="1996" height="160" alt="image" src="https://github.com/user-attachments/assets/56bdaaf4-f822-40f7-85d0-9dd6c2b8789a" />


Kodas buvo pamodifikuotas, kad uzrasytu modelio isejimo duomenis ant visu nuotrauku esanciu duotu duomenu aplanke ir tada issaugotu nuotraukas kitame rezultatu aplanke:
<img width="800" height="1380" alt="image" src="https://github.com/user-attachments/assets/ef90900c-4d16-4be5-9449-355879fab15b" />

Duotu nuotrauku rezultatai ikelti i "rezultatai" aplanka.

Taip pat prideti savi duomenys, paleistas kodas su savais duomenimis, taciau modelis neturejo pakankamai klasiu ir dave tik dvi klases visoms nuotraukoms, todel klasifikavimui panaudotas "google/vit-base-patch16-224" modelis ir rezultatai issaugoti "rezultatai_savi_duomenys" aplanke. Visgi ir sis modelis neturejo pakankamai klasiu.
