from prescritiva.text import normalize, stem, stems, tokenize


def test_normalize_remove_acento_e_caixa():
    assert normalize("Correção Angular") == "correcao angular"


def test_plural_e_singular_tem_o_mesmo_radical():
    for singular, plural in [
        ("polia", "polias"),
        ("correia", "correias"),
        ("rolamento", "rolamentos"),
        ("mancal", "mancais"),
        ("correcao", "correcoes"),
    ]:
        assert stem(singular) == stem(plural), f"{singular} != {plural}"


def test_derivacao_converge_para_o_mesmo_radical():
    assert stem("desalinhado") == stem("desalinhamento")
    assert stem("desbalanceado") == stem("desbalanceamento")
    assert stem("excentrico") == stem("excentricidade")


def test_termos_que_precisam_continuar_distintos():
    # "correia" x "correcao": todos os seis procedimentos tem "Correcao" no
    # titulo, entao juntar os dois faria a falha de correia casar com todos.
    assert stem("correia") != stem("correcao")
    assert stem("rotor") != stem("rotativo")
    assert stem("interno") != stem("externo")
    assert stem("ventoinha") != stem("ventilador")


def test_ocr_sem_acento_casa_com_texto_nativo():
    # O Doc1 vem de OCR e perde acentuacao; a busca precisa ser indiferente.
    assert stems("Correcao de Desalinhamento") == stems("Correção de Desalinhamento")


def test_tokenize_descarta_stopwords():
    assert "de" not in tokenize("procedimento de correcao")
