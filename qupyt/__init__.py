"""Entry point to qupyt"""
from qupyt.main import main

# ASCII art font doh.flf by Curtis Wanner
LOGO = """
                                                                                                         
                                                                                                         
     QQQQQQQQQ                       PPPPPPPPPPPPPPPPP                                     tttt          
   QQ:::::::::QQ                     P::::::::::::::::P                                 ttt:::t          
 QQ:::::::::::::QQ                   P::::::PPPPPP:::::P                                t:::::t          
Q:::::::QQQ:::::::Q                  PP:::::P     P:::::P                               t:::::t          
Q::::::O   Q::::::Quuuuuu    uuuuuu    P::::P     P:::::Pyyyyyyy           yyyyyyyttttttt:::::ttttttt    
Q:::::O     Q:::::Qu::::u    u::::u    P::::P     P:::::P y:::::y         y:::::y t:::::::::::::::::t    
Q:::::O     Q:::::Qu::::u    u::::u    P::::PPPPPP:::::P   y:::::y       y:::::y  t:::::::::::::::::t    
Q:::::O     Q:::::Qu::::u    u::::u    P:::::::::::::PP     y:::::y     y:::::y   tttttt:::::::tttttt    
Q:::::O     Q:::::Qu::::u    u::::u    P::::PPPPPPPPP        y:::::y   y:::::y          t:::::t          
Q:::::O     Q:::::Qu::::u    u::::u    P::::P                 y:::::y y:::::y           t:::::t          
Q:::::O  QQQQ:::::Qu::::u    u::::u    P::::P                  y:::::y:::::y            t:::::t          
Q::::::O Q::::::::Qu:::::uuuu:::::u    P::::P                   y:::::::::y             t:::::t    tttttt
Q:::::::QQ::::::::Qu:::::::::::::::uuPP::::::PP                  y:::::::y              t::::::tttt:::::t
 QQ::::::::::::::Q  u:::::::::::::::uP::::::::P                   y:::::y               tt::::::::::::::t
   QQ:::::::::::Q    uu::::::::uu:::uP::::::::P                  y:::::y                  tt:::::::::::tt
     QQQQQQQQ::::QQ    uuuuuuuu  uuuuPPPPPPPPPP                 y:::::y                     ttttttttttt  
             Q:::::Q                                           y:::::y                                   
              QQQQQQ                                          y:::::y                                    
                                                             y:::::y                                     
                                                            y:::::y                                      
                                                           yyyyyyy                                       
                                                                                                         
                                                                                                         
"""


def entry() -> None:
    """This function gets called when 'qupyt' is run
    from the commandline."""
    print(LOGO)
    main()
