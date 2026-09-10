#!/usr/bin/env python3
"""Standalone runner for the Czech tutoring quality tasks (interpretation, diagnosis, formulation).

Works against any OpenAI-compatible endpoint. Sampling is fixed to temperature 1.0 / top_p 0.95 /
max_tokens 64000 (the settings used on the wiki page). Extra request fields (reasoning control)
are passed with --extra as JSON, e.g. --extra '{"reasoning_effort": "low"}' or
--extra '{"chat_template_kwargs": {"thinking": false}}'.

Usage:
  python3 czech_tutor_eval.py --url http://localhost:8010/v1 --model deepseek-v4.1-flash \
      --extra '{"reasoning_effort": "low"}' --bench interpretace,diagnoza,formulace
Requires: httpx (pip install httpx); hunspell + cs_CZ dictionary for the typo rate (optional).
"""
import argparse, asyncio, json, re, statistics, subprocess, time
from pathlib import Path
import httpx

HERE = Path(__file__).parent
TEMPERATURE, TOP_P, MAX_TOKENS = 1.0, 0.95, 64000


class Client:
    def __init__(self, url, model, extra, conc):
        self.url, self.model, self.extra = url, model, extra
        self.sem = asyncio.Semaphore(conc)
        self.http = httpx.AsyncClient(base_url=url, timeout=1800)

    async def chat(self, messages, schema=None):
        body = {"model": self.model, "messages": messages, "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE, "top_p": TOP_P, **self.extra}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        async with self.sem:
            t0 = time.time()
            r = await self.http.post("/chat/completions", json=body)
            r.raise_for_status()
            j = r.json(); m = j["choices"][0]["message"]; u = j.get("usage") or {}
            content = (m.get("content") or "").strip()
            reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
            rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if rt is None:
                rt = max(0, int((u.get("completion_tokens") or 0) - len(content) / 3.3)) if reasoning else 0
            return dict(content=content, reasoning_tokens=int(rt), tokens=u.get("completion_tokens", 0),
                        dt=round(time.time() - t0, 2), truncated=j["choices"][0].get("finish_reason") == "length")


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


# ------------------------------------------------------------------ interpretation
INTERP_SCHEMA = {"type": "object", "properties": {
    "odpoved": {"type": ["string", "null"]},
    "zamer": {"type": "string", "enum": ["odpoved", "postup", "otazka_k_uloze", "napoveda", "vysvetleni",
                                         "nova_latka", "preskocit", "stop", "mimo_tema", "socialni", "jine"]},
    "emoce": {"type": "string", "enum": ["neutralni", "nejistota", "frustrace", "radost", "nuda_odpor"]}},
    "required": ["odpoved", "zamer", "emoce"], "additionalProperties": False}


def norm_ans(o):
    if o is None:
        return None
    s = str(o).strip()
    m = re.fullmatch(r"\(?([A-Ea-e])\)?[.:]?(\s.*)?", s)
    if m and (not m.group(2) or len(s) < 12):
        return m.group(1).upper()
    return re.sub(r"\s+|\.$", "", s.lower())


def ans_match(gold, pred):
    g, p = norm_ans(gold), norm_ans(pred)
    if g is None or p is None:
        return g is None and p is None
    return g == p or (len(g) >= 2 and (g in p or p in g))


async def bench_interpretace(cl, n):
    fx = json.loads((HERE / "interpretace.json").read_text())
    items = fx["polozky"][:n] if n else fx["polozky"]
    defs = "\n".join(f"- {k}: {v}" for k, v in fx["zamery"].items())
    system = ("Jsi interpretační vrstva AI mentora pro žáky 11–15 let. Dostaneš zprávu žáka poslanou "
              "během učení (typicky reakce na úlohu s možnostmi A–E nebo na otázku mentora). Urči:\n"
              "odpoved = k jaké odpovědi se žák přiklání (písmeno bez závorky, nebo hodnota), jinak null;\n"
              f"zamer = hlavní záměr zprávy:\n{defs}\n"
              "emoce = neutralni | nejistota | frustrace | radost | nuda_odpor.\nVrať pouze JSON.")

    async def one(p):
        r = await cl.chat([{"role": "system", "content": system},
                           {"role": "user", "content": f"Zpráva žáka: „{p['zprava']}“"}], INTERP_SCHEMA)
        try:
            j = json.loads(r["content"])
        except Exception:
            j = {}
        return dict(zamer_ok=j.get("zamer") in p["zamer_ok"], emoce_ok=j.get("emoce") == p["emoce"],
                    odpoved_ok=ans_match(p["odpoved"], j.get("odpoved")), **r)
    res = await asyncio.gather(*(one(p) for p in items))
    N = len(res)
    print("interpretace", json.dumps(dict(n=N, intent=round(100 * sum(r["zamer_ok"] for r in res) / N),
                                          answer=round(100 * sum(r["odpoved_ok"] for r in res) / N),
                                          emotion=round(100 * sum(r["emoce_ok"] for r in res) / N),
                                          reasoning_tokens_med=med([r["reasoning_tokens"] for r in res]),
                                          truncated=sum(r["truncated"] for r in res))))


# ------------------------------------------------------------------ diagnosis
async def bench_diagnoza(cl, n):
    fx = json.loads((HERE / "diagnoza.json").read_text())
    items = fx["polozky"][:n] if n else fx["polozky"]
    system = ("Jsi diagnostik chyb žáka (11–15 let, příprava na přijímačky). Dostaneš úlohu, "
              "to, co žák napsal, a kandidátní příčiny. Vyber JEDNU nejpravděpodobnější příčinu.")

    async def one(p):
        kand = "\n".join(f"- {k}: {fx['pricinny'][k]}" for k in p["kandidati"])
        schema = {"type": "object", "properties": {"pricina": {"type": "string", "enum": p["kandidati"]}},
                  "required": ["pricina"], "additionalProperties": False}
        r = await cl.chat([{"role": "system", "content": system},
                           {"role": "user", "content": f"ÚLOHA: {p['uloha']}\nŽÁK NAPSAL: {p['zak']}\nKANDIDÁTI:\n{kand}"}], schema)
        try:
            pred = json.loads(r["content"]).get("pricina")
        except Exception:
            pred = None
        return dict(ok=pred == p["spravne"], **r)
    res = await asyncio.gather(*(one(p) for p in items))
    print("diagnoza", json.dumps(dict(n=len(res), accuracy=round(100 * sum(r["ok"] for r in res) / len(res)),
                                      reasoning_tokens_med=med([r["reasoning_tokens"] for r in res]))))


# ------------------------------------------------------------------ formulation
PERSONA = ("Jsi Max, mentor pro žáky 11–15 let, kteří se připravují na přijímačky. Mluvíš česky, "
           "přátelsky a konkrétně, jako starší parťák, ne jako učitel u tabule. Nikdy neprozrazuješ "
           "výsledek úlohy, dokud ji žák sám nevyřeší. Nekáráš, nesrovnáváš s ostatními. "
           "Reaguješ na to, co žák právě napsal. Píšeš krátce.")


def brief_text(b):
    hist = "\n".join(f"{'MAX' if h['kdo'] == 'max' else 'ŽÁK'}: {h['text']}" for h in b["historie"])
    zak = ", ".join(b["zakazano"]) if b["zakazano"] else "nic"
    return (f"ŽÁK: {b['profil']}\nREŽIM: {b['rezim']}\nPOSLEDNÍ VÝMĚNA:\n{hist}\n\n"
            f"CO SE STALO / FAKTA (pravdivé, spočítal systém): {b['fakta']}\n"
            f"CO MÁ TVÁ REPLIKA UDĚLAT: {b['ukol']}\n"
            f"NESMÍ OBSAHOVAT: {zak}\nDÉLKA: nanejvýš {b['max_vet']} vět.\n"
            "Napiš jen repliku Maxe žákovi, nic jiného.")


WORD = re.compile(r"[A-Za-zÁ-žá-ž]{2,}")
WHITELIST = {"Max", "Maxi", "Maxe", "Kubo", "Kuba", "Emo", "Ema", "Tome", "Tom", "Báro", "Bára", "Šimone", "Šimon",
             "Adame", "Adam", "Lucie", "Lucko", "Minecraft", "Minecraftu", "CERMAT", "JPZ", "creeper", "ok", "OK", "jo", "jj"}


def typos(text):
    words = [w for w in WORD.findall(text) if w not in WHITELIST and not w.isupper() and not w[0].isupper()]
    if not words:
        return 0, 0
    try:
        p = subprocess.run(["hunspell", "-d", "cs_CZ", "-l"], input="\n".join(words), capture_output=True, text=True, timeout=30)
        bad = [w for w in p.stdout.split() if w and w not in WHITELIST]
    except Exception:
        return 0, len(words)
    return len(bad), len(words)


async def bench_formulace(cl, n, dump):
    fx = json.loads((HERE / "formulace.json").read_text())
    items = fx["polozky"][:n] if n else fx["polozky"]

    async def one(b):
        r = await cl.chat([{"role": "system", "content": PERSONA}, {"role": "user", "content": brief_text(b)}])
        t = r["content"]; tn = re.sub(r"\s+", " ", t)
        forbidden = [z for z in b["zakazano"] if z.lower() in tn.lower()]
        sentences = len([s for s in re.split(r"(?<=[.!?…])\s+", t.strip()) if s.strip()])
        bad, words = typos(t)
        return dict(id=b["id"], text=t, forbidden=forbidden, missing_question=b["otazka_povinna"] and "?" not in t,
                    too_long=sentences > b["max_vet"] + 1, meta=bool(re.search(r"\b(systém|nástroj|brief|instrukc|prompt)", t, re.I)),
                    typos=bad, words=words, **r)
    res = await asyncio.gather(*(one(b) for b in items))
    print("formulace", json.dumps(dict(n=len(res), forbidden_value=sum(bool(r["forbidden"]) for r in res),
                                       missing_question=sum(r["missing_question"] for r in res),
                                       too_long=sum(r["too_long"] for r in res), meta_talk=sum(r["meta"] for r in res),
                                       typos_per_100_words=round(100 * sum(r["typos"] for r in res) / max(1, sum(r["words"] for r in res)), 2),
                                       reasoning_tokens_med=med([r["reasoning_tokens"] for r in res]))))
    if dump:
        Path(dump).write_text(json.dumps(res, ensure_ascii=False, indent=1))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request (reasoning control)")
    ap.add_argument("--bench", default="interpretace,diagnoza,formulace"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--conc", type=int, default=4); ap.add_argument("--dump", default="", help="write formulation replies to this JSON file")
    a = ap.parse_args()
    cl = Client(a.url, a.model, json.loads(a.extra), a.conc)
    for b in a.bench.split(","):
        if b == "interpretace":
            await bench_interpretace(cl, a.n)
        elif b == "diagnoza":
            await bench_diagnoza(cl, a.n)
        elif b == "formulace":
            await bench_formulace(cl, a.n, a.dump)


if __name__ == "__main__":
    asyncio.run(main())
