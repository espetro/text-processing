# Generated from MinidownColor.g4 by ANTLR 4.7.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\r")
        buf.write("Q\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\3\2\3\2\3\2\6\2\16\n")
        buf.write("\2\r\2\16\2\17\3\2\5\2\23\n\2\3\2\5\2\26\n\2\3\3\3\3\3")
        buf.write("\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\5\3\"\n\3\3\4\3\4\3\4\3")
        buf.write("\4\3\4\3\4\7\4*\n\4\f\4\16\4-\13\4\3\5\3\5\3\5\3\5\3\5")
        buf.write("\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3")
        buf.write("\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5")
        buf.write("\5\5O\n\5\3\5\2\3\6\6\2\4\6\b\2\2\2T\2\25\3\2\2\2\4!\3")
        buf.write("\2\2\2\6#\3\2\2\2\bN\3\2\2\2\n\13\5\4\3\2\13\f\7\6\2\2")
        buf.write("\f\16\3\2\2\2\r\n\3\2\2\2\16\17\3\2\2\2\17\r\3\2\2\2\17")
        buf.write("\20\3\2\2\2\20\22\3\2\2\2\21\23\5\4\3\2\22\21\3\2\2\2")
        buf.write("\22\23\3\2\2\2\23\26\3\2\2\2\24\26\5\4\3\2\25\r\3\2\2")
        buf.write("\2\25\24\3\2\2\2\26\3\3\2\2\2\27\30\7\3\2\2\30\31\7\13")
        buf.write("\2\2\31\32\7\4\2\2\32\33\7\n\2\2\33\34\7\4\2\2\34\35\7")
        buf.write("\n\2\2\35\36\7\5\2\2\36\37\7\7\2\2\37\"\5\6\4\2 \"\5\6")
        buf.write("\4\2!\27\3\2\2\2! \3\2\2\2\"\5\3\2\2\2#$\b\4\1\2$%\5\b")
        buf.write("\5\2%+\3\2\2\2&\'\f\4\2\2\'(\7\7\2\2(*\5\b\5\2)&\3\2\2")
        buf.write("\2*-\3\2\2\2+)\3\2\2\2+,\3\2\2\2,\7\3\2\2\2-+\3\2\2\2")
        buf.write("./\7\3\2\2/\60\7\f\2\2\60\61\7\r\2\2\61\62\7\f\2\2\62")
        buf.write("\63\7\4\2\2\63\64\7\n\2\2\64\65\7\4\2\2\65\66\7\n\2\2")
        buf.write("\66O\7\5\2\2\678\7\3\2\289\7\f\2\29:\7\r\2\2:;\7\4\2\2")
        buf.write(";<\7\n\2\2<=\7\4\2\2=>\7\n\2\2>O\7\5\2\2?@\7\3\2\2@A\7")
        buf.write("\r\2\2AB\7\f\2\2BC\7\4\2\2CD\7\n\2\2DE\7\4\2\2EF\7\n\2")
        buf.write("\2FO\7\5\2\2GH\7\3\2\2HI\7\r\2\2IJ\7\4\2\2JK\7\n\2\2K")
        buf.write("L\7\4\2\2LM\7\n\2\2MO\7\5\2\2N.\3\2\2\2N\67\3\2\2\2N?")
        buf.write("\3\2\2\2NG\3\2\2\2O\t\3\2\2\2\b\17\22\25!+N")
        return buf.getvalue()


class MinidownColorParser ( Parser ):

    grammarFileName = "MinidownColor.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'('", "' , '", "')'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "NL", "WS", "HTML_TAG_IN", "HTML_TAG_OUT", "COLOR", 
                      "HEADER", "STYLE", "WORD" ]

    RULE_page = 0
    RULE_sentence = 1
    RULE_text = 2
    RULE_word = 3

    ruleNames =  [ "page", "sentence", "text", "word" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    NL=4
    WS=5
    HTML_TAG_IN=6
    HTML_TAG_OUT=7
    COLOR=8
    HEADER=9
    STYLE=10
    WORD=11

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class PageContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def sentence(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(MinidownColorParser.SentenceContext)
            else:
                return self.getTypedRuleContext(MinidownColorParser.SentenceContext,i)


        def NL(self, i:int=None):
            if i is None:
                return self.getTokens(MinidownColorParser.NL)
            else:
                return self.getToken(MinidownColorParser.NL, i)

        def getRuleIndex(self):
            return MinidownColorParser.RULE_page

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPage" ):
                listener.enterPage(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPage" ):
                listener.exitPage(self)




    def page(self):

        localctx = MinidownColorParser.PageContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_page)
        self._la = 0 # Token type
        try:
            self.state = 19
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 11 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 8
                        self.sentence()
                        self.state = 9
                        self.match(MinidownColorParser.NL)

                    else:
                        raise NoViableAltException(self)
                    self.state = 13 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,0,self._ctx)

                self.state = 16
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==MinidownColorParser.T__0:
                    self.state = 15
                    self.sentence()


                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 18
                self.sentence()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SentenceContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return MinidownColorParser.RULE_sentence

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class ParagraphContext(SentenceContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a MinidownColorParser.SentenceContext
            super().__init__(parser)
            self.value = None # TextContext
            self.copyFrom(ctx)

        def text(self):
            return self.getTypedRuleContext(MinidownColorParser.TextContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterParagraph" ):
                listener.enterParagraph(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitParagraph" ):
                listener.exitParagraph(self)


    class HeaderContext(SentenceContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a MinidownColorParser.SentenceContext
            super().__init__(parser)
            self.level = None # Token
            self.font = None # Token
            self.bg = None # Token
            self.value = None # TextContext
            self.copyFrom(ctx)

        def WS(self):
            return self.getToken(MinidownColorParser.WS, 0)
        def HEADER(self):
            return self.getToken(MinidownColorParser.HEADER, 0)
        def COLOR(self, i:int=None):
            if i is None:
                return self.getTokens(MinidownColorParser.COLOR)
            else:
                return self.getToken(MinidownColorParser.COLOR, i)
        def text(self):
            return self.getTypedRuleContext(MinidownColorParser.TextContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterHeader" ):
                listener.enterHeader(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitHeader" ):
                listener.exitHeader(self)



    def sentence(self):

        localctx = MinidownColorParser.SentenceContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_sentence)
        try:
            self.state = 31
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                localctx = MinidownColorParser.HeaderContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 21
                self.match(MinidownColorParser.T__0)
                self.state = 22
                localctx.level = self.match(MinidownColorParser.HEADER)
                self.state = 23
                self.match(MinidownColorParser.T__1)
                self.state = 24
                localctx.font = self.match(MinidownColorParser.COLOR)
                self.state = 25
                self.match(MinidownColorParser.T__1)
                self.state = 26
                localctx.bg = self.match(MinidownColorParser.COLOR)
                self.state = 27
                self.match(MinidownColorParser.T__2)
                self.state = 28
                self.match(MinidownColorParser.WS)
                self.state = 29
                localctx.value = self.text(0)
                pass

            elif la_ == 2:
                localctx = MinidownColorParser.ParagraphContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 30
                localctx.value = self.text(0)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TextContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def word(self):
            return self.getTypedRuleContext(MinidownColorParser.WordContext,0)


        def text(self):
            return self.getTypedRuleContext(MinidownColorParser.TextContext,0)


        def WS(self):
            return self.getToken(MinidownColorParser.WS, 0)

        def getRuleIndex(self):
            return MinidownColorParser.RULE_text

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterText" ):
                listener.enterText(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitText" ):
                listener.exitText(self)



    def text(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = MinidownColorParser.TextContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 4
        self.enterRecursionRule(localctx, 4, self.RULE_text, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 34
            self.word()
            self._ctx.stop = self._input.LT(-1)
            self.state = 41
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,4,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = MinidownColorParser.TextContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_text)
                    self.state = 36
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 37
                    self.match(MinidownColorParser.WS)
                    self.state = 38
                    self.word() 
                self.state = 43
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,4,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class WordContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.prefix = None # Token
            self.value = None # Token
            self.suffix = None # Token
            self.font = None # Token
            self.bg = None # Token

        def STYLE(self, i:int=None):
            if i is None:
                return self.getTokens(MinidownColorParser.STYLE)
            else:
                return self.getToken(MinidownColorParser.STYLE, i)

        def WORD(self):
            return self.getToken(MinidownColorParser.WORD, 0)

        def COLOR(self, i:int=None):
            if i is None:
                return self.getTokens(MinidownColorParser.COLOR)
            else:
                return self.getToken(MinidownColorParser.COLOR, i)

        def getRuleIndex(self):
            return MinidownColorParser.RULE_word

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterWord" ):
                listener.enterWord(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitWord" ):
                listener.exitWord(self)




    def word(self):

        localctx = MinidownColorParser.WordContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_word)
        try:
            self.state = 76
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 44
                self.match(MinidownColorParser.T__0)
                self.state = 45
                localctx.prefix = self.match(MinidownColorParser.STYLE)
                self.state = 46
                localctx.value = self.match(MinidownColorParser.WORD)
                self.state = 47
                localctx.suffix = self.match(MinidownColorParser.STYLE)
                self.state = 48
                self.match(MinidownColorParser.T__1)
                self.state = 49
                localctx.font = self.match(MinidownColorParser.COLOR)
                self.state = 50
                self.match(MinidownColorParser.T__1)
                self.state = 51
                localctx.bg = self.match(MinidownColorParser.COLOR)
                self.state = 52
                self.match(MinidownColorParser.T__2)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 53
                self.match(MinidownColorParser.T__0)
                self.state = 54
                localctx.prefix = self.match(MinidownColorParser.STYLE)
                self.state = 55
                localctx.value = self.match(MinidownColorParser.WORD)
                self.state = 56
                self.match(MinidownColorParser.T__1)
                self.state = 57
                localctx.font = self.match(MinidownColorParser.COLOR)
                self.state = 58
                self.match(MinidownColorParser.T__1)
                self.state = 59
                localctx.bg = self.match(MinidownColorParser.COLOR)
                self.state = 60
                self.match(MinidownColorParser.T__2)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 61
                self.match(MinidownColorParser.T__0)
                self.state = 62
                localctx.value = self.match(MinidownColorParser.WORD)
                self.state = 63
                localctx.suffix = self.match(MinidownColorParser.STYLE)
                self.state = 64
                self.match(MinidownColorParser.T__1)
                self.state = 65
                localctx.font = self.match(MinidownColorParser.COLOR)
                self.state = 66
                self.match(MinidownColorParser.T__1)
                self.state = 67
                localctx.bg = self.match(MinidownColorParser.COLOR)
                self.state = 68
                self.match(MinidownColorParser.T__2)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 69
                self.match(MinidownColorParser.T__0)
                self.state = 70
                localctx.value = self.match(MinidownColorParser.WORD)
                self.state = 71
                self.match(MinidownColorParser.T__1)
                self.state = 72
                localctx.font = self.match(MinidownColorParser.COLOR)
                self.state = 73
                self.match(MinidownColorParser.T__1)
                self.state = 74
                localctx.bg = self.match(MinidownColorParser.COLOR)
                self.state = 75
                self.match(MinidownColorParser.T__2)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[2] = self.text_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def text_sempred(self, localctx:TextContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 2)
         




