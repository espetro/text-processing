/* Parser and Lexer for reduced and coloured Markdown syntax in ANTLRv4.
 * based on https://daringfireball.net/projects/markdown/syntax
 *
 * This compiler is built for handwritten, reduced set of Markdown text, therefore classic HTML
 * tags are not included yet. PRs are welcomed.
 *
 * Note as well that '#', '@' and '$' are special characters, and are not yet allowed within text blocks. A proper
 * grammar should allow these characters to be used as normal characters.
 *
 * Compiles to ANTLR 4.7, generated lexer/parser for Python 3 target.
 *   antlr4 -Dlanguage=Python3 -listener -o grammar MinidownColor.g4
 *   (you must have a "grammar" folder within your root folder)
 *
 *
 * Word-color examples are created by using this function on common text (JS):
 *   text.split(" ").map(x => `(${x} , black , None)`).join(" ")
 * Note also the 2 whitespaces surrounding commas in the word-color tuple. This is done because the grammar
 * mistakes that comma as inside of the text token instead of the ',' parser token.
 * */

grammar MinidownColor;

/******** PARSER RULES ********/
/**
 * Parser rules are interpreted from top to bottom - rules defined first
 * will have higher precedence (eg. page > sentence, header > paragraph)
 */

page : (sentence NL)+ sentence?
     | sentence
     ;

sentence : '(' level=HEADER ' , ' font=COLOR ' , ' bg=COLOR ')' WS value=text    # header
         | value=text                                                            # paragraph
         ;

text: text WS word
    | word
    ;

word : '(' prefix=STYLE value=WORD suffix=STYLE ' , ' font=COLOR ' , ' bg=COLOR ')'
     | '(' prefix=STYLE value=WORD ' , ' font=COLOR ' , ' bg=COLOR ')'
     | '(' value=WORD suffix=STYLE ' , ' font=COLOR ' , ' bg=COLOR ')'
     | '(' value=WORD ' , ' font=COLOR ' , ' bg=COLOR ')'
     ;


/******** LEXER RULES ********/
/* lexer rules are analyzed in the order that they appear,
 * and they can be ambiguous, just like in JFlex.
 * see https://en.wikipedia.org/wiki/Latin_script_in_Unicode and
 *     https://github.com/antlr/antlr4/blob/master/doc/lexer-rules.md
 * for using UTF-8 charsets instead of ASCII charsets. Note that not all "Basic Latin" characters are available.
 */

NL: [\n\r]+;
WS: [\p{White_Space}];

HTML_TAG_IN: '<'[bi]'>';
HTML_TAG_OUT: '<'[bi]'/>';
COLOR: 'black' | 'blue' | 'brown' | 'green' | 'maroon' | 'mustard' | 'orange' | 'pink' | 'purple' | 'red' | 'yellow' | 'None';
HEADER: '#'+;
STYLE: [$@];
WORD: [\p{InLatin_1_Supplement}a-zA-Z0-9_.,:\-]+;

/* RESOURCES FOR UNDERSTANDING ANTLR4
 *   https://www.youtube.com/watch?v=pa8qG0I10_I
 *   https://github.com/duhai-alshukaili/antlr4
 *   https://medium.com/@raguiar2/building-a-working-calculator-in-python-with-antlr-d879e2ea9058
 *   https://github.com/antlr/antlr4/blob/master/doc/getting-started.md
 *   https://stackoverflow.com/questions/24299214/using-antlr-parser-and-lexer-separatly#24300037
 *   https://dexvis.wordpress.com/2012/11/22/a-tale-of-two-grammars/
 *   https://tomassetti.me/antlr-mega-tutorial/#creating-a-grammar
 */
