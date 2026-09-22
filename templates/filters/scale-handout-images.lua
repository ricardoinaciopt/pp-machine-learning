-- Apply a modest shared reduction to explicitly sized handout images.
-- Scaling both dimensions preserves the aspect ratio when both are supplied.
local function scaled_dimension(value, scale)
  if not value then return nil end
  local amount, unit = value:match('^([%d.]+)([%a%%]*)$')
  if not amount then return value end
  return string.format('%.6g%s', tonumber(amount) * scale, unit)
end

function Pandoc(doc)
  if not (FORMAT:match('latex') or FORMAT == 'docx') then return nil end
  local scale = tonumber(pandoc.utils.stringify(doc.meta['handout-image-scale'] or '1'))
  if not scale or scale <= 0 or scale > 1 then
    error('handout-image-scale must be greater than 0 and at most 1')
  end
  return doc:walk({Image = function(image)
    for _, dimension in ipairs({'width', 'height'}) do
      if image.attributes[dimension] then
        image.attributes[dimension] = scaled_dimension(image.attributes[dimension], scale)
      end
    end
    return image
  end})
end
