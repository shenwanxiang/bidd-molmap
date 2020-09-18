Donor = ["[N;!H0;v3,v4&+1]", "[O,S;H1;+0]", "[n&H1&+0]"]


Acceptor = ["[O,S;H1;v2;!$(*-*=[O,N,P,S])]",  "[O;H0;v2]", "[O,S;v1;-]", 
            "[N;v3;!$(N-*=[O,N,P,S])]", "[n&H0&+0]", "[o;+0;!$([o]:n);!$([o]:c:n)]"]


Positive = ["[#7;+]", "[N;H2&+0][$([C,a]);!$([C,a](=O))]", 
            "[N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]", 
            "[N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]"]

Negative = ["[C,S](=[O,S,P])-[O;H1,-1]"]

Hydrophobic = ["[C;D3,D4](-[CH3])-[CH3]", "[S;D2](-C)-C"]

Aromatic = ["a"]

pharmacophore_smarts = {"Donor": Donor, 
                         "Acceptor": Acceptor, 
                         "Positive":Positive,  
                         "Negative":Negative, 
                         "Hydrophobic":Hydrophobic,  
                         "Aromatic":Aromatic }