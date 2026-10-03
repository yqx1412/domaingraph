// D3 demo: walk from a concept to the sources that teach it, with timestamps.
// Run: docker exec -i domaingraph-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" < docs/d3-demo.cypher

// 1. Where is "AVL tree" taught? Concept -> chunk -> source, in time order.
MATCH (c:Concept)-[m:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(s:Source)
WHERE c.name_lc = 'avl tree' OR 'avl tree' IN c.aliases_lc
RETURN s.title AS source, ch.locator AS at, m.surfaces AS said_as
ORDER BY s.title, ch.start;

// 2. One hop further: what AVL trees use, and where each of those is explained.
MATCH (c:Concept {name_lc: 'avl tree'})-[r:RELATED_TO {predicate: 'uses'}]->(o:Concept)
      -[:MENTIONED_IN]->(ch:Chunk)-[:PART_OF]->(s:Source)
WITH o, r, s, ch ORDER BY ch.start
RETURN o.name AS uses, r.n_mentions AS times_said,
       s.title AS source, collect(ch.locator)[0..3] AS first_mentions
ORDER BY times_said DESC, uses
LIMIT 8;

// 3. Concepts taught in more than one lecture: the bridges between sources.
MATCH (c:Concept)-[:MENTIONED_IN]->(:Chunk)-[:PART_OF]->(s:Source)
WITH c, collect(DISTINCT s.title) AS sources, count(*) AS mentions
WHERE size(sources) > 1
RETURN c.name AS concept, sources, mentions
ORDER BY size(sources) DESC, mentions DESC
LIMIT 10;
