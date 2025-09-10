import React, { useState } from 'react'

export default function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [suggestions, setSuggestions] = useState([])

  async function searchProducts(q) {
    if (!q || q.length < 2) {
      setSuggestions([])
      return
    }
    try {
      const res = await fetch(`/api/products/search?q=${encodeURIComponent(q)}`)
      const data = await res.json()
      setSuggestions(data || [])
    } catch (e) {
      console.error(e)
    }
  }

  async function fetchScore(name) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`/api/score?product_name=${encodeURIComponent(name)}`)
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Request failed')
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function onSubmit(e) {
    e.preventDefault()
    if (query.trim()) {
      fetchScore(query.trim())
    }
  }

  return (
    <div style={{ maxWidth: 720, margin: '40px auto', fontFamily: 'system-ui, Arial, sans-serif' }}>
      <h1>Beauty Product Sentiment</h1>
      <p>Type a product name to estimate how favorable its reviews are.</p>
      <form onSubmit={onSubmit} style={{ display: 'flex', gap: 8 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
            searchProducts(e.target.value)
          }}
          placeholder="e.g., Glow Recipe Watermelon Dew Drops"
          style={{ flex: 1, padding: 10, borderRadius: 8, border: '1px solid #ccc' }}
        />
        <button type="submit" style={{ padding: '10px 16px', borderRadius: 8 }}>Analyze</button>
      </form>

      {suggestions.length > 0 && (
        <div style={{ marginTop: 10, border: '1px solid #eee', borderRadius: 8, padding: 8 }}>
          <strong>Suggestions:</strong>
          <ul>
            {suggestions.map((s, i) => (
              <li key={i} style={{ cursor: 'pointer' }} onClick={() => { setQuery(s.product_name); setSuggestions([]) }}>
                {s.product_name}{s.brand ? ` — ${s.brand}` : ''}
              </li>
            ))}
          </ul>
        </div>
      )}

      {loading && <p>Scoring...</p>}
      {error && <p style={{ color: 'crimson' }}>Error: {error}</p>}

      {result && (
        <div style={{ marginTop: 24, padding: 16, border: '1px solid #ddd', borderRadius: 12 }}>
          <h2>{result.product_name}</h2>
          <p><strong>Reviews used:</strong> {result.reviews_count}</p>
          <p><strong>Favorability:</strong> {result.favorable_percent}% ({result.verdict})</p>

          <div style={{ height: 16, background: '#eee', borderRadius: 8, overflow: 'hidden', marginTop: 8 }}>
            <div style={{ width: `${result.favorable_percent}%`, height: '100%' }} />
          </div>

          {result.sample_reviews?.length > 0 && (
            <details style={{ marginTop: 12 }}>
              <summary>Show sample reviews</summary>
              <ul>
                {result.sample_reviews.map((t, i) => <li key={i} style={{ margin: '8px 0' }}>{t}</li>)}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  )
}
