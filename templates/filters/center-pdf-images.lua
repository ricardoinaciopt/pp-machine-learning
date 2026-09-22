-- Center standalone migrated images in the available PDF text area.
-- Do not reinterpret inline formula images or change Word/PowerPoint objects.
local function center_images(block)
  if not FORMAT:match('latex') then return nil end
  local found = false
  for _, inline in ipairs(block.content) do
    if inline.t == 'Image' then
      found = true
      -- Respect an explicit local alignment choice.
      local align = inline.attributes['fig-align']
      if align and align ~= 'center' then return nil end
    elseif inline.t ~= 'Space' and inline.t ~= 'SoftBreak'
        and inline.t ~= 'LineBreak' then
      return nil
    end
  end
  if not found then return nil end
  return {
    pandoc.RawBlock('latex', '\\begingroup\\centering'),
    block,
    pandoc.RawBlock('latex', '\\par\\endgroup')
  }
end

return {{Para = center_images, Plain = center_images}}
