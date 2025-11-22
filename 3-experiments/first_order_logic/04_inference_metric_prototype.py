import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

# ================== CONFIGURAÇÕES ==================

BASE_DIR = Path("3-experiments/first_order_logic")

SCHEMA_FILE = "predicate_schema.json"
RULES_FILE = "logical_rules.json"

# Tipo para um fato: (nome_predicado, (arg1, arg2, ...))
Fact = Tuple[str, Tuple[str, ...]]

# ===================================================


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def facts_from_dict_list(fact_dicts: List[Dict]) -> Set[Fact]:
    """
    Converte uma lista de dicts do tipo
      {"predicate": "located_in", "args": ["A", "B"]}
    em um set de fatos do tipo:
      ("located_in", ("A", "B"))
    """
    facts: Set[Fact] = set()
    for fd in fact_dicts:
        pred = fd["predicate"]
        args = tuple(fd.get("args", []))
        facts.add((pred, args))
    return facts


def dict_list_from_facts(facts: Set[Fact]) -> List[Dict]:
    """
    Operação inversa de facts_from_dict_list, só para debug/impressão.
    """
    return [
        {"predicate": pred, "args": list(args)}
        for (pred, args) in sorted(facts)
    ]


# =============== MOTOR DE INFERÊNCIA ===============

def unify_premise_with_fact(premise: Dict, fact: Fact, env: Dict[str, str]) -> Dict[str, str] | None:
    """
    Tenta unificar uma premissa com um fato, respeitando um ambiente
    (env) pré-existente de variáveis -> constantes.

    premise: {"predicate": "located_in", "args": ["A", "B"]}
    fact:    ("located_in", ("Paris", "France"))
    env:     {"A": "Paris"}  (por exemplo)

    Retorna um novo env extendido se unificação for possível,
    ou None se falhar.
    """
    p_pred = premise["predicate"]
    p_args = premise.get("args", [])
    f_pred, f_args = fact

    # Predicado precisa ser igual
    if p_pred != f_pred:
        return None

    # Aridade deve bater
    if len(p_args) != len(f_args):
        return None

    new_env = dict(env)  # copia

    for var, const in zip(p_args, f_args):
        # Aqui assumimos que TODO arg na regra é variável (A, B, x, etc.)
        # e TODO arg no fato é constante (Paris, France, etc.)
        if var in new_env:
            # Variável já mapeada: precisa bater com o mesmo valor
            if new_env[var] != const:
                return None
        else:
            new_env[var] = const

    return new_env


def find_rule_matches(rule: Dict, facts: Set[Fact]) -> List[Dict[str, str]]:
    """
    Encontra todos os ambientes de variáveis (env) que satisfazem as
    premissas da regra com os fatos disponíveis.

    Retorna uma lista de dicts do tipo:
      [{"A": "Paris", "B": "France", "C": "Europe"}, ...]
    """
    premises = rule.get("premises", [])

    # Caso trivial: sem premissas (não usamos isso aqui)
    if not premises:
        return []

    # Estratégia de backtracking simples:
    # começamos com uma lista de envs possíveis (inicialmente [{}]),
    # e para cada premissa ampliamos essa lista com todos os matches.
    envs: List[Dict[str, str]] = [dict()]

    for prem in premises:
        new_envs: List[Dict[str, str]] = []

        for env in envs:
            # Tenta unificar esta premissa com TODOS os fatos, estendendo env
            for fact in facts:
                extended = unify_premise_with_fact(prem, fact, env)
                if extended is not None:
                    new_envs.append(extended)

        # Se nenhuma combinação funcionou para esta premissa, regra não dispara
        if not new_envs:
            return []

        envs = new_envs

    return envs


def apply_rule_once(rule: Dict, facts: Set[Fact]) -> Set[Fact]:
    """
    Aplica uma regra sobre o conjunto de fatos UMA VEZ,
    retornando o conjunto de NOVOS fatos inferidos (que ainda não
    estavam em 'facts').
    """
    new_facts: Set[Fact] = set()

    envs = find_rule_matches(rule, facts)
    conclusion = rule.get("conclusion", {})
    c_pred = conclusion.get("predicate")
    c_args_vars = conclusion.get("args", [])

    for env in envs:
        # Instancia a conclusão substituindo variáveis por constantes
        c_args_instantiated = []
        ok = True
        for var in c_args_vars:
            if var not in env:
                # variável da conclusão não foi ligada nas premissas:
                # descartamos esse env
                ok = False
                break
            c_args_instantiated.append(env[var])

        if not ok:
            continue

        fact_conclusion: Fact = (c_pred, tuple(c_args_instantiated))
        if fact_conclusion not in facts:
            new_facts.add(fact_conclusion)

    return new_facts


def forward_chaining(facts: Set[Fact], rules: List[Dict]) -> Set[Fact]:
    """
    Faz encadeamento direto até saturar (não há novos fatos).
    Retorna o closure (fatos originais + inferidos).
    """
    closure = set(facts)
    changed = True

    while changed:
        changed = False
        for rule in rules:
            inferred = apply_rule_once(rule, closure)
            if inferred:
                closure |= inferred
                changed = True

    return closure


# ========== CÁLCULO DAS MÉTRICAS LÓGICAS ==========

def logical_metrics(
    chunk_facts: Set[Fact],
    expl_facts: Set[Fact],
    rules: List[Dict],
    relevant_predicates: List[str] | None = None,
) -> Dict:
    """
    Calcula TP, FP, FN, precision, recall, F1 a partir de:
    - fatos do chunk,
    - fatos da explicação,
    - regras lógicas.

    relevant_predicates:
      Se None, considera todos os predicados no closure do chunk como relevantes.
      Caso contrário, considera apenas fatos cujo predicado esteja nessa lista.
    """
    # 1) Fecha a base do chunk via encadeamento direto
    closure_chunk = forward_chaining(chunk_facts, rules)

    # 2) Determina fatos relevantes no closure
    if relevant_predicates is None:
        relevant_closure = set(closure_chunk)
    else:
        relevant_closure = {
            f for f in closure_chunk if f[0] in relevant_predicates
        }

    # 3) TP = fatos da explicação que estão no closure relevante
    TP = expl_facts & relevant_closure

    # 4) FP = fatos da explicação que NÃO estão no closure relevante
    FP = expl_facts - relevant_closure

    # 5) FN = fatos relevantes do closure que NÃO estão na explicação
    FN = relevant_closure - expl_facts

    tp = len(TP)
    fp = len(FP)
    fn = len(FN)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "closure_chunk": closure_chunk,
        "relevant_closure": relevant_closure,
    }


# ===================== EXEMPLO =====================

def main():
    # Carrega schema e regras (mesmo que aqui o schema não seja usado diretamente)
    schema = load_json(SCHEMA_FILE)
    rules_obj = load_json(RULES_FILE)
    rules = rules_obj.get("rules", [])

    print("Loaded predicates:")
    for p in schema.get("predicates", []):
        print(f" - {p['name']}({', '.join(p['args'])})")

    print("\nLoaded rules:")
    for r in rules:
        print(f" - {r['name']}: {r['description']}")

    # Exemplo mínimo de fatos do chunk:
    # Chunk diz: "Paris is in France. France is in Europe."
    chunk_fact_dicts = [
        {"predicate": "located_in", "args": ["Paris", "France"]},
        {"predicate": "located_in", "args": ["France", "Europe"]},
    ]
    chunk_facts = facts_from_dict_list(chunk_fact_dicts)

    # Exemplo 1: explicação CORRETA (diz um fato inferível)
    expl1_fact_dicts = [
        {"predicate": "located_in", "args": ["Paris", "Europe"]},
    ]
    expl1_facts = facts_from_dict_list(expl1_fact_dicts)

    # Exemplo 2: explicação INCORRETA (diz algo não suportado)
    expl2_fact_dicts = [
        {"predicate": "located_in", "args": ["Paris", "Brazil"]},
    ]
    expl2_facts = facts_from_dict_list(expl2_fact_dicts)

    # Rodando métrica para expl1
    print("\n=== Example 1: correct explanation candidate ===")
    res1 = logical_metrics(
        chunk_facts,
        expl1_facts,
        rules,
        relevant_predicates=["located_in"],  # aqui consideramos só localização
    )

    print("TP:", dict_list_from_facts(res1["TP"]))
    print("FP:", dict_list_from_facts(res1["FP"]))
    print("FN:", dict_list_from_facts(res1["FN"]))
    print(f"precision={res1['precision']:.3f}, recall={res1['recall']:.3f}, f1={res1['f1']:.3f}")

    # Rodando métrica para expl2
    print("\n=== Example 2: incorrect explanation candidate ===")
    res2 = logical_metrics(
        chunk_facts,
        expl2_facts,
        rules,
        relevant_predicates=["located_in"],
    )

    print("TP:", dict_list_from_facts(res2["TP"]))
    print("FP:", dict_list_from_facts(res2["FP"]))
    print("FN:", dict_list_from_facts(res2["FN"]))
    print(f"precision={res2['precision']:.3f}, recall={res2['recall']:.3f}, f1={res2['f1']:.3f}")


if __name__ == "__main__":
    main()
