dtmc 
 
module GRU60000
s:[0..5] init 0; 
[]s=0 -> 1.0:(s'=1);
[]s=1 -> 1.0 :(s'=2);
[]s=2 -> 0.67 :(s'=4) + 0.33 :(s'=3);
[]s=3 -> 0.04148783977110158 :(s'=4) + 0.005722460658082976 :(s'=5) + 0.9527896995708155 :(s'=3);
[]s=4 -> 0.4517766497461929 :(s'=4) + 0.3350253807106599 :(s'=5) + 0.2131979695431472 :(s'=3);
[]s=5 -> 0.01818181818181818 :(s'=4) + 0.9696969696969697 :(s'=5) + 0.012121212121212121 :(s'=3);
endmodule 
